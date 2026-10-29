# Unified 3D Lumbar MRI Framework

This framework analyzes lumbar spine MRI studies containing sagittal and axial
DICOM series. It combines ROI detection, stenosis classification, anatomical
segmentation, and measurements, and provides the results through a FastAPI API.

The integrated [SwarmSpine chatbot](#9-run-the-chatbot-locally) provides a
browser interface for uploading studies, reviewing images and measurements,
and asking questions. Imaging inference and the chatbot use separate Python
environments. A local Ollama service supplies the language model.

| Local component | Port | Environment |
| --- | --- | --- |
| Imaging API | 8000 | `lss-framework` (or your existing `LSS`) |
| Chatbot | 8001 | `SwarmSpine_Inference/.venv` |
| Ollama | 11434 | Separate local service |

Open the chatbot at `http://127.0.0.1:8001`. All three services can run on the
same PC in WSL and communicate through localhost. No ngrok account, tunnel,
or public API address is required. Neither
imaging model weights nor LLM weights are included in this repository.

For the integrated workflow, complete sections **1–4**, then **9**. Upload
the study through the chatbot once both servers and Ollama are running.
Sections 5 and 7 describe alternative API-only workflows; they are not extra
steps required to use the chatbot.

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
| `SwarmSpine_Inference/` | Chatbot, image routing, configuration template, and offline checks. |
| `local_models/` | Optional local LLM storage; entirely excluded from Git. |

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

If you have the example study separately, place it at this repository location
(DICOM data and ZIP archives are excluded from Git):

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

Keep this terminal running and continue to section 9 for the chatbot.
To use the API directly instead, open [the API interface](http://127.0.0.1:8000/docs)
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

## 8. Local connection between components

The integrated project runs on one PC:

```text
Browser -> chatbot: http://127.0.0.1:8001
Chatbot -> imaging API: http://127.0.0.1:8000/predict
Chatbot -> local LLM: http://127.0.0.1:11434
Image retrieval -> imaging API: http://127.0.0.1:8000/static/...
```

The chatbot sends the study to the local API and uses its returned image URLs.
Keep the API, chatbot, and Ollama running in the same Linux/WSL distribution.
Only open port 8001 in the browser for the integrated interface; there is no
separate upload step through the API documentation page.

## 9. Run the chatbot locally

Complete sections 1–4 first to install the imaging environment, place the
trained checkpoints, and start the API. Then follow the steps below to set up
the chatbot and its local language model. All commands are for Linux/WSL.

### Services

| Service | Default address | Purpose |
| --- | --- | --- |
| Imaging API | http://127.0.0.1:8000 | DICOM processing and output images |
| Chatbot | http://127.0.0.1:8001 | Browser interface and study questions |
| Ollama | http://127.0.0.1:11434 | Local language model |

Use the same Linux/WSL distribution for these services. Python packages do not
install the Ollama service or its model.

### First-time chatbot setup

From the repository root:

```bash
cd SwarmSpine_Inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Keep these local defaults in `.env`:

```env
SPINESWARM_API_URL=http://127.0.0.1:8000/predict
SPINESWARM_PORT=8001
OLLAMA_API_BASE=http://127.0.0.1:11434
SPINESWARM_MODEL=gemma3:12b
```

Configuration loads from this folder's `.env`; existing process environment
variables take precedence. Put comments on separate lines. Do not commit `.env`.
Optional OpenAI support uses your own `OPENAI_API_KEY` and a supported model;
it sends the study context to that provider instead of running entirely locally.

### Install and prepare the local LLM

Install Ollama in Linux/WSL using its [installation guide](https://docs.ollama.com/linux).
For example:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
ollama pull gemma3:12b
curl http://127.0.0.1:11434/api/tags
```

If systemd is unavailable, run `ollama serve` in a separate terminal and leave
it open before pulling the model. Do not start a second server on an occupied
port. The tags response should list `gemma3:12b`; it confirms availability,
not successful generation or sufficient memory. Model and imaging workloads
share machine resources. The model selector can use another installed model.

Weights remain in Ollama's local storage, outside Git. To use this repository's
ignored `local_models/ollama/` directory instead, follow
[local model storage instructions](#10-optional-local-model-storage).
No model download, installation, or service startup is performed automatically
by this repository.

### Start the project

Terminal 1, from the repository root, after placing the imaging weights:

```bash
conda activate lss-framework
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Terminal 2, from the repository root:

```bash
cd SwarmSpine_Inference
source .venv/bin/activate
python check_connection.py
python -m spine_swarm.web_app
```

Keep Ollama running too. Open [the chatbot](http://127.0.0.1:8001).
If using the author's already configured backend environment, activate `LSS`
instead of `lss-framework`; do not reinstall a working imaging environment.

Upload a DICOM ZIP to run imaging inference and the clinical panel. Upload a
saved API `results.json` to skip imaging inference; loading it still runs the
panel. The image files referenced by that JSON must remain available through
the configured API. Generic questions and reports invoke the language model;
typed image lookups read the loaded findings without an LLM call.

### Image behavior

| Request | Image source |
| --- | --- |
| Sagittal classification | Sagittal classification Grad-CAM image |
| Axial classification | `output/classification/axial_combined/` only |
| Axial detection | `output/ROIs_detection/axial/axial_estimation_debug/` only |
| Sagittal detection | Sagittal ROI detection overlay |
| Measurement | Matching measurement image and stored values |
| Segmentation | Segmentation overlay, not the original image |

Typed responses show at most one image and do not replace missing images with
another category. Axial combined images show multiple regions at a given level.
Highest-grade sagittal requests select one available image among tied records
in record order and disclose ties. Captions use clinical wording rather than
internal field names. Sagittal classification images have level labels; new
sagittal measurement images omit embedded top headings.

The backend now emits the actual axial combined file in `gradcam_path` and
`gradcam_image` (`path` and `url`). The adapter also handles older responses
with a valid entry-level combined image URL. Detection URLs remain in the API
JSON but are selected only for detection requests.

### Example questions

- Show one sagittal image of the worst foraminal stenosis and state its level, side, and severity.
- Show the axial foraminal classification at L1-L2.
- Show the axial detection image at L4-L5.
- Show the lumbar lordosis measurement and its image.
- Show the spondylolisthesis measurements and corresponding image.
- Show the vertebral deformity measurements.
- Show the axial segmentation at L4-L5.

### Troubleshooting

| Symptom | What to check |
| --- | --- |
| LLM connection refused | Ollama is running; `OLLAMA_API_BASE` is reachable from the chatbot environment |
| Model not found | `ollama list` includes the exact selected model tag |
| API connection failure | Backend is running; `.env` points to its `/predict` endpoint |
| Image unavailable | Matching study outputs still exist; URL host matches `SPINESWARM_API_URL` |
| Address already in use | Stop the old process before restarting that service |
| Old labels or image titles | Restart backend and regenerate images; code changes do not edit saved PNGs |

An old remote URL in saved JSON is not automatically rewritten to localhost.
Use results from the local API. Windows-hosted Ollama may require a different
reachable host address from WSL. Changes require restarting the affected service.

One active study is held in memory. A new upload clears the previous context;
restarting clears it too. The imaging API replaces `input/` and `output/` on
each upload. Use one study/user at a time and preserve needed outputs separately.

### Checks you can run

```bash
python -B check_integration.py
python check_connection.py
```

The first command uses offline fixtures/mocks. The second makes a read-only
GET request for the API contract; it does not run inference or call an LLM.
The chatbot's `/health` endpoint only checks that its web server responds.
These checks do not validate clinical accuracy or model performance.
Dependency ranges are not a tested lockfile.

The existing backend stage timings cover imaging processing, not chatbot or
LLM response time. No evaluation datasets, scores, saved reports, credentials,
Python environments, or trained weights are included in this chatbot folder.

## 10. Optional local model storage

The entire `local_models/` directory is excluded from Git. Create it only if
you want a custom local LLM storage location. Imaging weights must remain at
the exact paths in `checkpoints/manifest.json`; do not move them here.

For Ollama, the simplest option is to keep its existing default model storage.
There is no need to redownload a model that the running server already lists.

If you specifically want local LLM weights under this project, use a manually
started Ollama server. From the repository root in Linux/WSL:

```bash
mkdir -p local_models/ollama
export OLLAMA_MODELS="$(pwd)/local_models/ollama"
ollama serve
```

First stop any existing Ollama server on port 11434 (for a systemd service,
`sudo systemctl stop ollama`). Leave the manual-server terminal open. In another
terminal, download to that server's configured storage:

```bash
ollama pull gemma3:12b
ollama list
```

Set `OLLAMA_MODELS` each time you manually start that server. Setting it in the
chatbot `.env` does not configure a separate Ollama process or system service.
Changing model storage does not move existing downloaded models automatically.
These are instructions only; no weights or model server are included here.


## Citation

If you use this framework, the associated code, or datasets in your research, please cite the following works:

```bibtex
@article{Abdulmahmod2026SagittalMRI,
  title   = {Medical Spine Sagittal MRI Dataset for Segmentation and Foraminal Stenosis Detection},
  author  = {Abdulmahmod, Osamah F. and Al-antari, Mugahed A. and Kwon, Hyunwook and others},
  journal = {Scientific Data},
  volume  = {13},
  pages   = {809},
  year    = {2026},
  doi     = {10.1038/s41597-026-07138-x}
}

@dataset{Kwon2026AxialMRI,
  title     = {Medical Lumbar Spine 3D Axial MRI Dataset for Stenosis Detection, Severity Classification, and Anatomical Segmentation},
  author    = {Kwon, Hyunwook and Abdulmahmod, Osamah F. and Habib, Afnan and Raza, Mukhlis and Bütün, Ertan and Talo, Muhammed and Kaplan, Metin and Ertuğrul, Bilal and Al-antari, Mugahed A.},
  publisher = {Mendeley Data},
  version   = {V1},
  year      = {2026},
  doi       = {10.17632/wjxkrkxhb8.1}
}

@inproceedings{Abdulmahmod2025LumbarStenosis,
  title     = {A Novel Lumbar Foraminal Stenosis Detection and Classification Framework Using 3D Sagittal MRI Dataset},
  author    = {Abdulmahmod, Osamah F. and Habib, Afnan and Raza, Mukhlis and Gu, Yeong Hyeon and Talo, Muhammed and Al-antari, Mugahed A.},
  booktitle = {2025 9th International Artificial Intelligence and Data Processing Symposium (IDAP)},
  pages     = {1--7},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/IDAP68205.2025.11222352}
}
```

We appreciate your acknowledgment of our work. Please cite the relevant publications if you use, modify, or build upon this repository.


