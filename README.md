# Unified 3D Lumbar MRI Framework

An inference framework for lumbar spine MRI studies, combining sagittal and
axial DICOM processing, ROI detection, stenosis classification, segmentation,
and anatomical measurements. A FastAPI service accepts a study ZIP and returns
JSON results and links to generated images.

This release contains the inference pipeline through the HTTP API. Training
datasets, patient studies, generated results, and model weights are excluded.
It processes slices from volumetric studies; the name does not imply every
network is a 3D model. Intended for research use.

## Pipeline

1. Extract the ZIP and identify axial and sagittal DICOM slices.
2. Select sagittal slices and detect ROIs using YOLO and DINO ensembles.
3. Segment anatomy and project disc levels to axial slices.
4. Detect and estimate axial ROIs and segment axial anatomy.
5. Classify foraminal, lateral recess, and central canal stenosis as supported
   by each branch; compute anatomical measurements.
6. Export visualizations, results, and stage timings for L1-L2 through L5-S1.

## Repository layout

```text
api.py                  FastAPI entry point
lss_pipeline.py         Study orchestration and JSON export
tools/                  DICOM, ROI, segmentation, classification, measurements
checkpoints/             Model configurations and expected-weight manifest
examples/predict.py     Python HTTP client
scripts/check_setup.py  Weight/import/CUDA checks
stage_timing.py         Per-stage timing
summarize_timings.py    Timing aggregation
```

## Installation

Use Linux or WSL2 with an NVIDIA GPU. The bundled DINO extension requires CUDA
at build time. Use Python 3.10, an NVIDIA driver supporting the selected CUDA
runtime, a matching CUDA toolkit with `nvcc`, and a C++ build toolchain. Native
Windows and CPU-only execution have not been validated for this release.

Clone your copy of this repository and enter its root directory:

```bash
git clone https://github.com/YOUR_USERNAME/Unified-3D-Lumbar-MRI-Framework.git
cd Unified-3D-Lumbar-MRI-Framework
conda create -n lss-framework python=3.10 -y
conda activate lss-framework
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-cuda.txt
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username. The CUDA
12.8 PyTorch command follows the [official version instructions](https://pytorch.org/get-started/previous-versions/).
Check `nvidia-smi`, `nvcc --version`, and `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`
before compiling. Set `CUDA_HOME` to your toolkit directory if it is not detected.
The toolkit and PyTorch CUDA versions must be compatible.

Compile the included DINO operator from this repository:

```bash
cd tools/roi/DINO/models/dino/ops
python -m pip install --no-build-isolation .
cd ../../../../../..
python -m pip check
```

See the [upstream DINO installation notes](https://github.com/IDEA-Research/DINO#installation)
for its custom operator. Rebuild it after changing PyTorch or CUDA. The supplied
requirements replace local Conda build paths from the original environment;
a fresh GPU installation and full inference still need validation on your hardware.

## Windows, WSL2, and file paths

Run the installation commands and inference server in a Linux terminal or in
Ubuntu/another Linux distribution under WSL2. On Windows, open your WSL2 terminal
for these commands. Native Windows inference has not been validated.

Use paths appropriate to the terminal running the command:

| Where the command runs | Example study path |
| --- | --- |
| WSL2, reading Windows drive D | `/mnt/d/data/study.zip` |
| WSL2, reading Windows drive C | `/mnt/c/Users/YOUR_NAME/data/study.zip` |
| Native Linux | `/home/YOUR_NAME/data/study.zip` |
| Windows PowerShell, HTTP client only | `D:\data\study.zip` |

For example, `D:\data\study.zip` in Windows is `/mnt/d/data/study.zip` inside
WSL2. Replace the path with the location on your own computer. `/mnt/d/` is a
WSL drive mapping, not a required project folder or a general Linux convention.

If an example is included at `examples/data/example_study.zip`, run this from
the repository root in the inference environment:

```bash
python examples/predict.py examples/data/example_study.zip
```

That relative path needs no `/mnt/` prefix. For a study elsewhere on drive D:

```bash
python examples/predict.py /mnt/d/data/study.zip
```

The HTTP client can also run in Windows PowerShell while the inference server
runs in WSL2. In that case, use Windows paths for the client upload:

```powershell
curl.exe --fail-with-body -X POST http://127.0.0.1:8000/predict -F "file=@D:\data\study.zip" -o response.json
```

The client sends the ZIP contents to the API; the server does not need access
to the client's original file path. Paths returned in JSON refer to the server;
use the returned HTTP asset URLs to retrieve images from a different computer.

## Model weights (required)

Weights are distributed separately and are not downloaded automatically.
**A public weight download URL has not yet been provided.** Obtain the matching
trained weights from the project authors before running inference. Repository
owners should add their weight-release link here before announcing availability.

Place all files at the paths listed in [checkpoints/manifest.json](checkpoints/manifest.json).
Preserve spelling and case, including `Sagital_AGMH` and `sagittal_Dino`.
The JSON configurations are included; do not substitute unrelated checkpoints.

```bash
python scripts/check_setup.py
```

This checks files, imports, and CUDA availability, but does not verify model
compatibility or prediction quality.

## Prepare an input study

Use one de-identified lumbar MRI study per ZIP, with axial and sagittal DICOM
series from the same study. Folder names containing `axial` or `sagittal` guide
routing; otherwise the code uses DICOM orientation metadata where available.

```text
study.zip
  axial/
    slice001.dcm
    slice002.dcm
  sagittal/
    slice001.dcm
    slice002.dcm
```

The ZIP may contain additional nesting. Files must be readable by pydicom and
include the orientation, position, and pixel spacing metadata needed by the
pipeline. PNG/JPEG images are not replacements for these DICOM inputs. Compressed
DICOM transfer syntaxes may require additional pydicom pixel decoder packages.
Input sequences and image characteristics should match those used to develop
the models; arbitrary MRI protocols are not guaranteed to work.

## Run the local API

Run from the repository root:

```bash
conda activate lss-framework
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000/docs, expand `POST /predict`, select **Try it out**,
choose the ZIP in the `file` field, and click **Execute**.

Alternatively, submit from a terminal:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/study.zip" -o response.json
```

On Windows PowerShell use `curl.exe` and your Windows study path. A Python client
is also included:

```bash
python examples/predict.py /path/to/study.zip --output response.json
```

Inference can take time, especially during the first request when models load.
Submit one study at a time. All runs use the same `input/` and `output/` folders:
**each upload deletes the previous input and output contents**. Download or copy
results before submitting another study. Do not run multiple workers, concurrent
clients, or direct pipeline calls at the same time.

## Results and errors

The response includes `status`, `run_id`, `input`, `sagittal`, `axial`,
`result_json`, and `timings` on successful runs. Inspect the actual JSON for the
branch-specific predictions and measurements; fields vary with available results.

- `output/results.json`: saved response.
- `output/`: generated overlays and visualization assets.
- `/static/...`: HTTP access to saved output files; local filesystem paths are
  paths on the server, not on the client.
- `output/timings.json`: latest timing record.
- `benchmark/stage_timings.jsonl`: persistent timing history.

Axial measurements include level-specific entries such as `L1-L2_herniation`
with `disc_herniation`, `herniation_ratio`, and overlay references. ROI URLs are
intentionally omitted by the existing serializer; other assets may include URLs.
Asset URLs stop representing a previous study once a new upload replaces output.

Non-ZIP filenames return HTTP 400. Unhandled request errors return HTTP 500.
Errors caught inside the pipeline can return HTTP 200 with `"status": "failed"`
and an `error` object. Always inspect `status`; HTTP success alone does not mean
inference completed. Detailed errors may include local paths and tracebacks.

## Use the pipeline from Python

From the repository root, in the installed inference environment:

```python
from pathlib import Path
from lss_pipeline import run_zip

result = run_zip(Path("/path/to/study.zip"), filename="study.zip")
print(result["status"])
```

Direct use has the same shared-folder replacement behavior as the API. Without
`base_url`, output references are local paths rather than public asset URLs.

## Share an API endpoint

After confirming local inference, a temporary ngrok tunnel can expose the local
service (install ngrok and configure your own account first):

```bash
ngrok http 8000
```

Use its HTTPS forwarding address as the client base URL:

```bash
python examples/predict.py /path/to/study.zip --url https://YOUR-TUNNEL.ngrok.app
```

Swagger UI is at `https://YOUR-TUNNEL.ngrok.app/docs`. Keep both the server and
tunnel running. This API currently has no authentication, permits all CORS
origins, serves output files publicly to anyone who can reach the endpoint,
and has no application-level upload size limit. Use only controlled research
demonstrations with de-identified data. Before persistent or multi-user hosting,
add authentication, HTTPS/access controls, request limits, and isolated per-job
storage. A tunnel URL is not an API key; no third-party API key is required for
the local inference pipeline.

## Timing and checks

```bash
python -m unittest test_stage_timing.py
python summarize_timings.py
```

The existing unit tests cover timing/export behavior without loading models.
The summarizer writes `benchmark/timing_summary.json`. These are not model
accuracy tests. GPU inference requires all matching weights and a study.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing checkpoint | Follow `checkpoints/manifest.json`; check filename case. |
| Missing `MultiScaleDeformableAttention` | Build the bundled operator in the active environment. |
| CUDA unavailable/build failure | Check driver, toolkit, `CUDA_HOME`, compiler, and PyTorch build. |
| Missing `mamba_ssm` | Install `requirements-cuda.txt` after PyTorch with build isolation disabled. |
| Model state-dict mismatch | Use weights and inference configurations from the same release. |
| Empty or failed study | Check axial/sagittal inputs and required DICOM geometry metadata. |
| GPU out of memory | Stop other GPU workloads and process one study at a time. |
| Earlier results disappear | Copy the output before the next request. |

## Publish to GitHub

Before publishing, add the authors' chosen `LICENSE`, the required third-party
license notices, the actual repository URL, a model-weight release URL, and the
paper citation when available. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
No paper metadata or permission to choose a license was assumed during packaging.

Create an empty repository on GitHub, then run from this directory:

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Initial inference framework and API release"
git remote add origin https://github.com/YOUR_USERNAME/Unified-3D-Lumbar-MRI-Framework.git
git push -u origin main
```

Review the staged files before committing. `.gitignore` excludes weights, DICOM,
ZIPs, patient input/output, caches, and benchmark history. Publish approved model
weights separately and link them above. No GitHub upload is performed by preparing
this folder.
