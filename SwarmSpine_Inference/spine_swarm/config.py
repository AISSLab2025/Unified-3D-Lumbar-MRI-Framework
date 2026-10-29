"""Configuration shared by the web application and API connection."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Simple KEY=value format; existing process environment takes precedence.
env_file = BASE_DIR / ".env"
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)

API_URL = os.getenv("SPINESWARM_API_URL", "http://127.0.0.1:8000/predict")
API_TIMEOUT = float(os.getenv("SPINESWARM_API_TIMEOUT", "900"))
HOST = os.getenv("SPINESWARM_HOST", "127.0.0.1")
PORT = int(os.getenv("SPINESWARM_PORT", "8001"))
DEFAULT_MODEL = os.getenv("SPINESWARM_MODEL", "gemma3:12b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip("/")
# Compatibility constant for shared helper imports; the web app uses memory only.
UPLOADED_JSON_PATH = str(BASE_DIR / "runtime" / "uploaded_data.json")
