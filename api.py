import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from lss_pipeline import OUTPUT_ROOT, PROJECT_ROOT, run_zip


os.chdir(PROJECT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="LSS Lumbar Spine Inference API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(OUTPUT_ROOT)), name="static")


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file containing the DICOM study.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_zip_path = Path(tmp.name)

    try:
        result = run_zip(
            zip_path=tmp_zip_path,
            filename=file.filename,
            base_url=str(request.base_url),
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc
    finally:
        try:
            tmp_zip_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
