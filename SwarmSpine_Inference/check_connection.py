"""Read-only API contract check: GET only, never uploads or invokes an LLM."""
import sys
from urllib.parse import urlsplit, urlunsplit
import requests
from spine_swarm.config import API_URL


def main():
    url = urlsplit(API_URL)
    base = urlunsplit((url.scheme, url.netloc, "", "", ""))
    try:
        response = requests.get(base + "/openapi.json", timeout=10)
        response.raise_for_status()
        contract = response.json()
        operation = contract.get("paths", {}).get(url.path, {}).get("post", {})
        content = operation.get("requestBody", {}).get("content", {})
        if "multipart/form-data" not in content:
            raise ValueError("Configured endpoint does not declare a multipart POST upload.")
        print("PASS: API is reachable and declares the ZIP upload interface.")
        print("No prediction or chatbot generation was requested.")
        return 0
    except (requests.RequestException, ValueError) as exc:
        print(f"Connection check failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
