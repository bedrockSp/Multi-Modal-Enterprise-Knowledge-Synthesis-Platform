# Multi-Modal-Enterprise-Knowledge-Synthesis-Platform

# Windows CPU Quick Start

If you're on a Windows box without an NVIDIA GPU, use the CPU profile. After installing the prereqs below (**Python 3.11**, Node 22, MongoDB, Tesseract, Ollama for Windows):

```powershell
# from the repo root
.\setup.ps1 all        # creates venv, installs CPU requirements, pulls models, pre-warms HF cache, starts Ollama
.\setup.ps1 backend    # in one terminal
.\setup.ps1 frontend   # in another terminal
```

`setup.ps1` has individual subcommands too: `venv`, `set-models`, `ollama`, `ollama-stop`, `doctor`, `backend`, `frontend`.

### Python 3.11 is required, not 3.12+

Several ML dependencies (sentence-transformers, easyocr, torch combos) don't ship wheels for Python 3.12, 3.13, or 3.14 yet. `setup.ps1` detects 3.11 via the Windows Python launcher (`py -3.11`). If you previously created `virtualEnv\` with a newer Python, delete the folder and re-run `.\setup.ps1 venv`.

### Corporate network / SSL certificate errors

If you hit `[SSL: CERTIFICATE_VERIFY_FAILED]` when models download from HuggingFace, your corporate proxy is inspecting HTTPS with a custom CA that Python's bundled certifi doesn't trust. The CPU requirements include [`pip-system-certs`](https://pypi.org/project/pip-system-certs/) which patches Python's SSL stack to use the Windows certificate store (where corporate CAs are installed). Make sure it's installed in the venv that runs `backend.py`:

```powershell
.\virtualEnv\Scripts\python.exe -m pip install pip-system-certs
.\setup.ps1 doctor    # pre-downloads embedding + cross-encoder models
```

`doctor` triggers an explicit, one-time HuggingFace download so the first `backend.py` boot doesn't crash on a model that hasn't been cached yet. The embedding model loads at module import time, so a failed download is a fatal startup error.

### What the CPU profile does

- Installs `torch` from the CPU wheel index (`requirements-windows-cpu.txt`) — no CUDA libs are downloaded.
- Skips `python-igraph` (no Windows wheels); the codebase falls back to NetworkX for community detection automatically.
- Defaults `EMBEDDING_DEVICE=cpu`, `CROSS_ENCODER_DEVICE=cpu`, `EASYOCR_GPU=false` in `.env` (see `.env.example`).
- Lowers OCR worker counts to laptop-safe values (EasyOCR: 2, Tesseract: half of `os.cpu_count()`, GLM-OCR: 1).
- Pulls `qwen2.5:7b` instead of the default `gpt-oss:20b` — the 20B model is unusably slow on most CPUs. Set `MAIN_MODEL=qwen2.5:7b` in your `.env` to match.
- Auto-detects Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### Performance expectations on CPU

OCR is ~5-10× slower, embedding ~10× slower, cross-encoder rerank ~4-9× slower than the GPU path. Ingestion of a 50-page PDF can take several minutes. Queries should still respond within ~30s for a 7B model on a modern CPU.

### If you have a CUDA GPU on Windows

Skip this section and use `requirements.txt` as usual; set `EMBEDDING_DEVICE=auto` (default), `EASYOCR_GPU=true`, and keep `MAIN_MODEL=gpt-oss:20b-50k-8k`.

---

# Prerequisites

- **Node.js**
- **Tesseract**
- **MongoDB**
- **Python**

## Node.js

install from https://nodejs.org/dist/v22.20.0/node-v22.20.0-x64.msi

a msi file will be downloaded. Run the installer and follow the prompts to complete the installation.

- check this option when installing

![alt text](assistance/node.png)

#### Verify installation

```bash
node -v
npm -v
```

if `npm -v` error's out try restarting your terminal / pc / vscode

# Tesseract

- Download the Windows installer from https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe

- run the installer

# MongoDB

install from https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-8.2.1-signed.msi

- run the installer and follow the prompts to complete the installation.

- an application named "MongoDB Compass" will be installed in your system. open it and follow steps to connect to your local db

- do next next next when prompted between complete and custom select complete then next next next install

![alt text](assistance/mongo3.png)

- add new connection

![alt text](assistance/mongo1.png)

- save and connect

![alt text](assistance/mongo2.png)

# Python

- Download the Windows installer from https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

- run the installer and follow check the following option when prompted
- check both options, in image one is unchecked
- please make sure you installl python from the given link to avoid clashes with other versions of python

![alt text](assistance/python.png)

## _Please read `core/constant.py`_

- line no 13
  - REMOTE_GPU = True, # Use remote GPU LLMs
  - default value is true set it to false to use local gpu servers
  - set it to False to use local ollama servers

# Setting Up the Server's

### run this script if you are in samsung's virtual machine

- please use power shell to run this script (right side terminal in vscode)
- it handels error related to ssl certification in node, git, python and huggingface
  ![alt text](assistance/ter.png)

```powershell
prism_vm.ps1
```

### 1. clone the repository

```bash
git clone https://github.com/bugslayer01/Knowledge-Synthesis-Platform.git
```

navigate into the project directory:

```bash
cd Knowledge-Synthesis-Platform
```

Follow these steps to set up and run the backend server:

### 2. Create a Python Virtual Environment (Please use python 3.11 for this)

It's recommended to use a virtual environment to isolate dependencies:

```bash
python -m venv virtualEnv
```

#### Activate the virtual environment:

- **Windows (PowerShell):**

```powershell
.\virtualEnv\Scripts\activate.ps1
```

- **Windows (Terminal):**

```powershell
.\virtualEnv\Scripts\activate
```

- **Linux / macOS:**(depend on your shell)

```bash
source virtualEnv/bin/activate
```

### 2. Install Dependencies

Make sure you have `pip` updated, then install the required packages:

```bash
pip install -r requirements.txt

```

### 3. Please rename .env.example to .env file in project root

### 4. create models and run 2 instances of ollama for parallel processing

- Open Terminal
  ```bash
  OLLAMA_HOST=0.0.0.0:11435 OLLAMA_KEEP_ALIVE=-1 ollama serve
  ```
- Open new terminal
  ```bash
   OLLAMA_HOST=0.0.0.0:11434 OLLAMA_KEEP_ALIVE=-1 ollama serve
  ```
- load and make custome models
  - if you are running this for the first time please run the following command to setup the models
  ```bash
  ./setmodel.sh
  ```
- to change parameters of models plese edit these files
  - for gpt oss
  - run this to apply changes
    ```bash
    ./setmodel.sh
    ```

### 4. Start the Server

run the FastAPI server:
make sure that virtual environment is activated

```bash
python backend.py
```

The server will start at:

```
http://0.0.0.0:3000
```

# Frontend Setup

Follow these steps to set up and run the frontend server:

open a new terminal and run frontend.py

```bash
python frontend.py
```
