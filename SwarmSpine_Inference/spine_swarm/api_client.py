"""HTTP adapter to the existing LSS API. Importing never runs inference."""
import requests
from .config import API_URL, API_TIMEOUT


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The inference API must return a JSON object.")
    if payload.get("status") == "failed" or payload.get("error"):
        raise ValueError("The inference API reported failure; no study was loaded.")
    if not any(isinstance(payload.get(key), dict) and payload[key]
               for key in ("sagittal", "axial", "Sagittal_ROI_classification",
                           "Sagittal_measurements_and_diagnosis")):
        raise ValueError("Response contains no supported study data.")
    return payload


def predict_zip(filename, content):
    if not filename.lower().endswith(".zip") or not content:
        raise ValueError("A nonempty DICOM ZIP is required.")
    response = requests.post(
        API_URL,
        files={"file": (filename, content, "application/zip")},
        headers={"Accept": "application/json", "ngrok-skip-browser-warning": "1"},
        timeout=(10, API_TIMEOUT),
    )
    response.raise_for_status()
    return validate_payload(response.json())
