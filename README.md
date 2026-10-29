# Unified 3D Lumbar MRI Framework

Run lumbar MRI inference and the chatbot together on your PC. The chatbot
sends the uploaded study to the local imaging API and uses the returned
findings and images. Ollama runs the language model. No ngrok is required.

Use **Ubuntu/WSL2 or Linux** for all commands below. Backend installation
requires Conda, Python 3.10, an NVIDIA GPU, a driver compatible with CUDA 12.8,
the CUDA toolkit with `nvcc`, and a C++ compiler.

## 1. Install the backend once

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

Obtain the trained imaging weights from the project authors and place them at
the paths in [checkpoints/manifest.json](checkpoints/manifest.json). Weights are
not included in GitHub. Then check the installation:

```bash
python scripts/check_setup.py
```

If your existing `LSS` environment already runs the backend, keep using it
instead of creating `lss-framework` again.

## 2. Install the chatbot once

From the repository root:

```bash
cd SwarmSpine_Inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set these values in `SwarmSpine_Inference/.env`:

```env
SPINESWARM_API_URL=http://127.0.0.1:8000/predict
SPINESWARM_HOST=127.0.0.1
SPINESWARM_PORT=8001
OLLAMA_API_BASE=http://127.0.0.1:11434
SPINESWARM_MODEL=gemma3:12b
```

`SPINESWARM_API_URL` connects the chatbot to the imaging backend.
`OLLAMA_API_BASE` connects it to the local language model.

## 3. Install and prepare Ollama once

In Ubuntu/WSL or Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
ollama pull gemma3:12b
curl http://127.0.0.1:11434/api/tags
```

The last command should list `gemma3:12b`. If systemd is unavailable, start
`ollama serve` in another terminal and leave it open before downloading the
model. Do not start a second server if Ollama is already running.
The LLM is downloaded locally; it is not included in this repository.

## 4. Start the backend and chatbot

Keep both terminals in the same Linux/WSL distribution, and keep Ollama running.
Open each terminal in the repository root.

**Terminal 1 — imaging backend:**

```bash
conda activate lss-framework
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Use `conda activate LSS` instead if that is your existing working environment.
Leave this terminal running.

**Terminal 2 — chatbot:**

```bash
cd SwarmSpine_Inference
source .venv/bin/activate
python check_connection.py
python -m spine_swarm.web_app
```

The connection check reads the API contract without running inference.
Open **[http://127.0.0.1:8001](http://127.0.0.1:8001)** in your browser.

## 5. Run inference through the chatbot

1. Select **gemma3:12b**.
2. Upload a DICOM ZIP containing the study's axial and sagittal series.
3. Click **Run SpineSwarm Analysis**.
4. Wait for processing to finish, then review the report and ask questions.

The connection is:

```text
Chatbot (8001) -> imaging API /predict (8000) -> findings and images
Chatbot (8001) -> Ollama (11434) -> clinical panel responses
```

Images are retrieved through the local API at
`http://127.0.0.1:8000/static/...`. There is no separate upload through the API
page and no public tunnel to configure.

You can upload a saved API `results.json` to skip imaging inference; the
chatbot still runs its clinical panel, and the referenced images must remain
available. Process one study at a time: each ZIP upload replaces the backend's
previous input and output files.

If you see **connection refused**, ensure the corresponding service is
running: backend on `8000`, chatbot on `8001`, and Ollama on `11434`.

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


