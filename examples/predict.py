"""Submit one study: python examples/predict.py study.zip --url http://127.0.0.1:8000"""
import argparse
import json
from pathlib import Path
import requests

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("response.json"))
    args = parser.parse_args()
    with args.study.open("rb") as stream:
        response = requests.post(args.url.rstrip("/") + "/predict",
                                 files={"file": (args.study.name, stream, "application/zip")},
                                 timeout=(30, 3600))
    response.raise_for_status()
    result = response.json()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; pipeline status: {result.get('status')}")
    if result.get("status") != "success":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
