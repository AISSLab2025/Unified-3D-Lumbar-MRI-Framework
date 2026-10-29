import json
import logging
import os
import sys
import base64
import requests
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Add parent directory to sys.path to enable direct execution
current_dir = Path(__file__).resolve().parent
if current_dir.name == "spine_swarm":
    sys.path.insert(0, str(current_dir.parent))

from spine_swarm.langchain_agents import LangChainClinicalPanel
from spine_swarm.agents import (
    MODERATOR_PROMPT, MODERATOR_CHAT_PROMPT,
    get_patient_context_str, find_images_in_json, get_image_links_context
)

logger = logging.getLogger("SpineSwarm.WebApp")
logging.basicConfig(level=logging.INFO)

from spine_swarm.config import UPLOADED_JSON_PATH, OLLAMA_API_BASE, API_URL, HOST, PORT, DEFAULT_MODEL
from spine_swarm.api_client import predict_zip, validate_payload
BASE_DIR = Path(__file__).resolve().parent.parent

# Simple state storage for the current loaded session
current_pipeline_data = {}

class SpineSwarmHTTPHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        # Serve the single-page HTML frontend
        if self.path in ("/", "/index.html"):
            html_file = Path(__file__).resolve().parent / "index.html"
            try:
                with open(html_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except Exception as e:
                self.send_error(500, f"Error reading index.html: {e}")
        elif self.path == "/api/models":
            # Return available models (both Ollama local and OpenAI standard)
            models = []
            
            # Fetch local Ollama models
            try:
                logger.info(f"Fetching local Ollama models from {OLLAMA_API_BASE}/api/tags...")
                response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    for m in data.get("models", []):
                        name = m.get("name")
                        if name:
                            models.append({"id": name, "name": f"Ollama: {name}", "type": "ollama"})
            except Exception as e:
                logger.warning(f"Could not connect to Ollama to list models: {e}")
                
            # If no Ollama models found, add default gemma3:12b as placeholder
            if not any(m["type"] == "ollama" for m in models):
                models.append({"id": "gemma3:12b", "name": "Ollama: gemma3:12b (unverified; Ollama unavailable or no models found)", "type": "ollama"})
                
            # Add OpenAI ChatGPT models
            models.append({"id": "gpt-5.4-mini", "name": "ChatGPT: gpt-5.4-mini", "type": "openai"})
            models.append({"id": "gpt-4o", "name": "ChatGPT: gpt-4o", "type": "openai"})
            models.append({"id": "gpt-4o-mini", "name": "ChatGPT: gpt-4o-mini", "type": "openai"})
            models.append({"id": "gpt-3.5-turbo", "name": "ChatGPT: gpt-3.5-turbo", "type": "openai"})
            
            if not any(m["id"] == DEFAULT_MODEL for m in models):
                models.insert(0, {"id": DEFAULT_MODEL, "name": DEFAULT_MODEL, "type": "configured"})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": models, "default_model": DEFAULT_MODEL}).encode('utf-8'))
        elif self.path.startswith("/api/image"):
            self.serve_proxied_image()
        else:
            self.send_error(404, "File not found")

    def serve_proxied_image(self):
        """Fetch a diagnostic image from the prediction API and stream it to the page.

        The prediction API is published through an ngrok free tunnel, which answers
        browser-looking requests with an HTML interstitial instead of the file. A
        page cannot set the header that skips it, so the fetch is made here and the
        bytes are relayed to the <img> tag.
        """
        query = urllib.parse.urlparse(self.path).query
        src = urllib.parse.parse_qs(query).get("src", [""])[0]
        if not src:
            self.send_error(400, "Missing src parameter")
            return

        parsed = urllib.parse.urlparse(src)
        if parsed.scheme not in ("http", "https"):
            self.send_error(400, "Unsupported URL scheme")
            return

        # Only relay from the configured prediction API host, never an arbitrary URL
        allowed_host = urllib.parse.urlparse(API_URL).netloc
        if parsed.netloc != allowed_host:
            logger.warning(f"Refused image proxy for unexpected host: {parsed.netloc}")
            self.send_error(403, "Host not allowed")
            return

        try:
            resp = requests.get(
                src,
                headers={
                    # Skips the ngrok free-tier browser interstitial
                    "ngrok-skip-browser-warning": "1",
                    "User-Agent": "SpineSwarm/1.0",
                    "Accept": "image/*",
                },
                timeout=30,
                stream=True,
            )
        except Exception as e:
            logger.error(f"Image proxy request failed for {src}: {e}")
            self.send_error(502, "Could not reach the image host")
            return

        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code != 200:
            logger.warning(f"Image proxy upstream returned {resp.status_code} for {src}")
            self.send_error(resp.status_code if resp.status_code in (404, 403) else 502,
                            "Image not available upstream")
            return
        if not content_type.startswith("image/"):
            logger.warning(f"Image proxy got {content_type!r} instead of an image for {src}")
            self.send_error(502, "Upstream did not return an image")
            return

        body = resp.content
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global current_pipeline_data
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {e}"}).encode('utf-8'))
            return

        if self.path == "/api/upload":
            current_pipeline_data = {}
            filename = body.get("filename", "")
            file_content_str = body.get("filecontent", "")
            symptoms = body.get("symptoms", "")
            if not symptoms or not symptoms.strip():
                symptoms = "No symptoms provided."
                
            if not file_content_str:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty file content"}).encode('utf-8'))
                return

            try:
                # Check if upload is a ZIP file
                is_zip = False
                if filename and filename.lower().endswith(".zip"):
                    is_zip = True
                elif file_content_str.startswith("data:application/zip;base64,") or file_content_str.startswith("data:application/x-zip-compressed;base64,") or file_content_str.startswith("data:application/octet-stream;base64,"):
                    is_zip = True

                if is_zip:
                    logger.info("Decoding base64 ZIP payload...")
                    if "," in file_content_str:
                        base64_data = file_content_str.split(",")[1]
                    else:
                        base64_data = file_content_str
                    zip_bytes = base64.b64decode(base64_data)

                    # Forward ZIP to SpineSwarm deep learning model prediction API
                    logger.info(f"Forwarding ZIP archive to prediction API: {API_URL}...")
                    parsed_json = predict_zip(filename or 'patient_scan.zip', zip_bytes)
                else:
                    # Parse uploaded JSON outputs directly
                    parsed_json = json.loads(file_content_str)
                    logger.info("Parsed uploaded JSON outputs successfully.")

                # Normalize uploaded JSON outputs to handle both legacy and updated API response schemas
                from spine_swarm.agents import normalize_patient_data
                validate_payload(parsed_json)
                normalized_json = normalize_patient_data(parsed_json)

                # Initialize clinical panel debate
                selected_model = body.get("model", None)
                api_key = body.get("api_key", None)
                
                panel = LangChainClinicalPanel(pipeline_data=normalized_json, model_name=selected_model, api_key=api_key)
                report, debate_nodes = panel.run_debate()
                pipeline_summary = "Ingestion and clinical analysis completed successfully."

                # Store pipeline data in memory for subsequent chat queries
                current_pipeline_data = normalized_json
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "pipeline_summary": pipeline_summary,
                    "report": report,
                    "transcript": "",
                    "debate_nodes": debate_nodes,
                    "pipeline_data": normalized_json
                }).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error handling upload: {e}", exc_info=True)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Failed to execute pipeline: {e}"}).encode('utf-8'))

        elif self.path == "/api/chat":
            user_message = body.get("message", "")
            history = body.get("history", [])
            symptoms = body.get("symptoms", "No symptoms provided.")
            
            if not user_message:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty query"}).encode('utf-8'))
                return
            
            try:
                active_data = current_pipeline_data
                if not active_data:
                    raise ValueError("No active study. Upload a study successfully before chatting.")
                patient_data_str = get_patient_context_str(active_data)
                image_links_str = get_image_links_context(active_data)
                
                # Clean history to exclude the current user message at the end to avoid duplication
                history_clean = history[:-1] if (history and history[-1].get("role") == "user") else history
                
                # Execute debate and consensus compilation using the LangChain clinical panel
                selected_model = body.get("model", None)
                api_key = body.get("api_key", None)
                panel = LangChainClinicalPanel(pipeline_data=active_data, model_name=selected_model, api_key=api_key)
                reply, chat_debate_nodes = panel.run_chat_query(user_message, history_clean)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "reply": reply,
                    "debate_nodes": chat_debate_nodes
                }).encode('utf-8'))
            except Exception as e:
                logger.error(f"Error handling chat: {e}", exc_info=True)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Failed to generate response: {e}"}).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")


class SpineSwarmHTTPServer(HTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port another process is
    # already serving. The new process then looks like it started fine while the
    # stale one keeps answering requests with the old code still loaded in memory.
    # Only enable address reuse where it is safe (POSIX TIME_WAIT handling).
    allow_reuse_address = (os.name != "nt")


def run():
    server_address = (HOST, PORT)
    try:
        httpd = SpineSwarmHTTPServer(server_address, SpineSwarmHTTPHandler)
    except OSError as e:
        logger.error(
            f"Cannot bind port {PORT}: {e}\n"
            f"Another SpineSwarm server is already running. Stop it first, "
            f"otherwise it will keep serving the previously loaded code."
        )
        sys.exit(1)
    logger.info(f"SpineSwarm Web App running on http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Web Server...")
        httpd.server_close()

if __name__ == "__main__":
    run()
