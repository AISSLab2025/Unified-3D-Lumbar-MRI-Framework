# Unified 3D Lumbar MRI Framework

This framework analyzes lumbar spine MRI studies containing sagittal and axial
DICOM series. It combines ROI detection, stenosis classification, anatomical
segmentation, and measurements, and provides the results through a FastAPI API.

## How the code works

`api.py` accepts a study ZIP through `POST /predict` and passes it to
`run_zip()` in `lss_pipeline.py`. The pipeline:

1. Extracts the DICOM files and separates sagittal and axial slices.
2. Selects sagittal slices and detects regions of interest using YOLO and DINO.
3. Classifies sagittal foraminal stenosis and segments sagittal anatomy.
4. Uses sagittal anatomy to locate disc levels in the axial series.
5. Detects axial ROIs, estimates missing ROIs, and segments axial anatomy.
6. Computes measurements and classifies axial foraminal, lateral recess, and
   central canal stenosis at lumbar disc levels L1-L2 through L5-S1.
7. Saves JSON results, visualization images, and processing times.

| File or directory | Purpose |
| --- | --- |
| `api.py` | HTTP upload endpoint and access to output images. |
| `lss_pipeline.py` | Runs the complete study analysis and prepares JSON results. |
| `tools/dicom/` | Reads DICOM data and identifies image volumes. |
| `tools/slice_selection/` | Selects sagittal slices and projects disc levels to axial slices. |
| `tools/roi/` | Detects ROIs with YOLO/DINO and estimates missing axial ROIs. |
| `tools/classification/` | Predicts stenosis classes for the detected ROIs. |
| `tools/segmentation/` | Produces anatomical segmentation masks. |
| `tools/measurement/` | Calculates anatomical measurements from images and masks. |
| `checkpoints/` | Model configuration files and the required-weight manifest. |
| `examples/predict.py` | Sends a study ZIP to a running API and saves its response. |
| `stage_timing.py` | Records processing times for pipeline stages. |

## 1. Install the environment

Run the installation and inference server in **Linux or a WSL2 Linux terminal**.
Use Python 3.10, an NVIDIA GPU, an NVIDIA driver compatible with CUDA 12.8,
a matching CUDA toolkit with `nvcc`, and a C++ compiler. The bundled DINO
operator is compiled locally against PyTorch and CUDA.

```bash
git clone https://github.com/AISSLab2025/Unified-3D-Lumbar-MRI-Framework.git
cd Unified-3D-Lumbar-MRI-Framework

conda create -n lss-framework python=3.10 -y
conda activate lss-framework

python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-cuda.txt

cd tools/roi/DINO/models/dino/ops
python -m pip install --no-build-isolation .
cd ../../../../../..
```

Native Windows inference has not been validated. Windows users should run these
commands inside WSL2, such as an Ubuntu terminal.

## 2. Add the trained models

Obtain the matching trained model weights from the project authors and place
them at the paths listed in [checkpoints/manifest.json](checkpoints/manifest.json).
The repository includes model configurations; trained weights are distributed
separately. A public weight download link is not currently provided.

Keep all directory names and filename capitalization exactly as listed,
including `Sagital_AGMH` and `sagittal_Dino`.

Check the required files and installed modules:

```bash
python scripts/check_setup.py
```

## 3. Prepare a study ZIP

Use one study containing both sagittal and axial DICOM series. For example:

```text
study.zip
├── axial/
│   ├── slice001.dcm
│   └── slice002.dcm
└── sagittal/
    ├── slice001.dcm
    └── slice002.dcm
```

Folder names containing `axial` and `sagittal` guide slice routing. Otherwise,
the code uses DICOM orientation metadata where available. Keep the DICOM
orientation, position, and pixel spacing information needed for measurements.

For the anonymized example study, use this repository location:

```text
examples/data/0004.zip
```

### Which file path should I use?

Run commands from the repository root. A relative path such as
`examples/data/0004.zip` works in Linux and WSL2 without a `/mnt/` prefix.
Use `/mnt/` when accessing a Windows drive from WSL2:

| Command environment | Example path |
| --- | --- |
| Linux or WSL2, example inside the repository | `examples/data/0004.zip` |
| WSL2, study on Windows drive D | `/mnt/d/data/0004.zip` |
| Native Linux, study outside the repository | `/home/user/data/0004.zip` |
| Windows PowerShell, HTTP upload client | `D:\data\0004.zip` |

For example, Windows `D:\data\0004.zip` becomes `/mnt/d/data/0004.zip` in WSL2.

## 4. Start the API

From the repository root, run:

```bash
conda activate lss-framework
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep this terminal running. Open [the API interface](http://127.0.0.1:8000/docs)
in your browser:

1. Expand **POST /predict**.
2. Select **Try it out**.
3. Choose your study ZIP in the `file` field.
4. Select **Execute** to run the analysis and view the JSON response.

## 5. Submit a study from the command line

Open a second terminal, activate the environment, and enter the repository root.
Submit the example study:

```bash
conda activate lss-framework
python examples/predict.py examples/data/0004.zip --output response.json
```

For a different study stored on Windows drive D, run this inside WSL2:

```bash
python examples/predict.py /mnt/d/data/my_study.zip --output response.json
```

Alternatively, upload using curl from Linux or WSL2:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -F "file=@examples/data/0004.zip" -o response.json
```

An HTTP upload client can also run in Windows PowerShell while the server runs
in WSL2. Use the Windows path in that terminal:

```powershell
curl.exe --fail-with-body -X POST http://127.0.0.1:8000/predict -F "file=@D:\data\0004.zip" -o response.json
```

## 6. Read the results

The client saves the API response as `response.json`. The server also saves:

| Output | Contents |
| --- | --- |
| `output/results.json` | Study results, classifications, measurements, and image references. |
| `output/` | Generated segmentation overlays and other visualization images. |
| `output/timings.json` | Processing times for the latest study. |
| `benchmark/stage_timings.jsonl` | Processing-time history across runs. |

In a completed response, `sagittal` and `axial` contain the corresponding
analysis results, and `status` indicates whether the pipeline completed.
Image references may contain a server-local `path` and an HTTP `url`.
Open the HTTP URL to retrieve an image through the API's `/static/` route.

**Process one study at a time. Each upload replaces the contents of `input/`
and `output/`. Copy the results you want to keep before uploading another study.**

## 7. Use the pipeline directly in Python

To run without starting the API, execute this from the repository root in the
installed inference environment:

```python
from pathlib import Path
from lss_pipeline import run_zip

result = run_zip(
    zip_path=Path("examples/data/0004.zip"),
    filename="0004.zip",
)

print(result["status"])
print(result["result_json"])
```

This uses the same pipeline and output folders as the API. Without a server
base URL, output image references use local filesystem paths.

## 8. Share the API

With the local API running, open another terminal and start ngrok using your
installed and configured ngrok account:

```bash
ngrok http 8000
```

Copy its HTTPS forwarding address and use it as the client base URL:

```bash
python examples/predict.py examples/data/0004.zip \
  --url https://YOUR-TUNNEL.ngrok.app --output response.json
```

Users can also open `https://YOUR-TUNNEL.ngrok.app/docs` and upload a ZIP through
the API interface. Keep the API server and ngrok running while the endpoint is
in use. The current API has no authentication; share it only with intended
research users and submit studies sequentially.
