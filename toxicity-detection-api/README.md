# Toxicity Detection API

A FastAPI service that checks a user prompt before it reaches an LLM.

```
query → detection (safe / toxic) → classification (only if toxic) → result
```

- **Detection:** a fine-tuned toxicity model (trained on ToxicChat). Returns safe or toxic.
- **Classification:** an ensemble of two `BAAI/bge-base-en-v1.5` models (trained on JailbreakBench). Picks 1 of 10 risk categories.

The models are hosted on the Hugging Face Hub and download automatically on first run. You don't need to download them yourself.

| Model | Hub repo |
|---|---|
| Detection | `jueedeshmukh/01_toxic_detection_daniel` |
| Classification | `jueedeshmukh/toxic_prompt_classification_riya` |

---

## Files in this folder

| File | What it does |
|---|---|
| `main.py` | The API. Defines the `/check` endpoint. |
| `pipeline.py` | Loads the models and runs detection → classification. |
| `requirements.txt` | Python libraries, pinned to tested versions. |
| `Dockerfile` | Instructions for building the Docker image. |
| `.dockerignore` | Files Docker should skip when building. |

---

## Choose how to run it

- **Option A: Docker.** Recommended if you have Docker Desktop. Same setup on every machine, no Python version issues.
- **Option B: Plain Python.** Use this if you don't have Docker.

Both give you the same API. Pick one.

---

## Option A: Run with Docker

### Before you start

- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Open Docker Desktop and wait until it says it's running.
- About 4 GB of free disk space.

### Step 1: Open a terminal in this folder

```bash
cd toxicity-detection-api
```

### Step 2: Build the image

```bash
docker build -t prompt-gateway .
```

This follows the `Dockerfile` to build an image with Python, CPU-only PyTorch, and all libraries installed. The first build takes several minutes. Later builds are faster because Docker caches each step.

### Step 3: Start the container

```bash
docker run -p 7860:7860 prompt-gateway
```

`-p 7860:7860` connects port 7860 on your computer to port 7860 inside the container.

On startup, the container downloads all three models (~1.4 GB). Wait until you see:

```
Application startup complete.
```

### Step 4: Open the test page

Go to **http://localhost:7860/docs**, then follow [Testing the API](#testing-the-api) below.

### Stopping it

Press `Ctrl + C` in the terminal.

---

## Option B: Run with plain Python

### Before you start

- **Python 3.12** is recommended. Versions above 3.13 may fail to install `scikit-learn==1.6.1`. Check yours with:
  ```bash
  python --version
  ```
- About 4 GB of free disk space.

### Step 1: Open a terminal in this folder

```bash
cd toxicity-detection-api
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

This makes an isolated Python environment for this project, so its library versions don't clash with anything else on your computer. You only do this once.

### Step 3: Activate it

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal line. Activate it every time you open a new terminal to work on this project.

### Step 4: Install PyTorch (CPU-only)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The CPU version is much smaller than the default GPU version, and the API doesn't need a GPU.

### Step 5: Install everything else

```bash
pip install -r requirements.txt
```

### Step 6: Start the server

```bash
uvicorn main:app --reload
```

- `main:app` means "the `app` variable inside `main.py`."
- `--reload` restarts the server automatically when you save a code change.

The first run downloads the models (~1.4 GB). Wait until you see:

```
Application startup complete.
```

### Step 7: Open the test page

Go to **http://127.0.0.1:8000/docs**, then follow [Testing the API](#testing-the-api) below.

### Stopping it

Press `Ctrl + C` in the terminal.

---

## Testing the API

1. Open the `/docs` page (port **7860** for Docker, **8000** for Python).
2. Click the green **POST /check** row.
3. Click **Try it out**.
4. Replace the example body with:
   ```json
   {"text": "You are a worthless idiot and everyone hates you."}
   ```
5. Click **Execute**.

### What you should see

A toxic prompt:
```json
{
  "label": "unsafe",
  "category": "Harassment/Discrimination",
  "confidence": 0.39
}
```

A safe prompt (try `{"text": "What is the capital of France?"}`):
```json
{
  "label": "safe"
}
```

`confidence` is the classifier's probability for its top category, from 0 to 1. With 10 categories, a random guess would be about 0.10.

### The 10 categories

Disinformation, Economic harm, Expert advice, Fraud/Deception, Government decision-making, Harassment/Discrimination, Malware/Hacking, Physical harm, Privacy, Sexual/Adult content.

### Calling it from code

```python
import requests

r = requests.post("http://localhost:7860/check", json={"text": "your prompt here"})
print(r.json())
```

Use port `8000` instead of `7860` if you're running Option B.

---

## Troubleshooting

**`{"detail": "Not Found"}` when opening the link in the terminal.**
The server is working. The terminal links to `/`, which has no page. Add `/docs` to the end of the URL.

**`{"detail": "Not Found"}` at `/docs/check`.**
Stop at `/docs`. `/check` is a row inside that page, not a separate URL.

**`Method Not Allowed` when visiting `/check` in the browser.**
The address bar sends a GET request, but `/check` only accepts POST. Use the `/docs` page or the code example above.

**Warning: "You are sending unauthenticated requests to the HF Hub."**
Harmless. The model repos are public, so no token is needed. To make it go away:
- **Python:** run `hf auth login` once and paste a Hugging Face token (Read permission is enough).
- **Docker:** `docker run -e HF_TOKEN=hf_yourtoken -p 7860:7860 prompt-gateway`

Never put a token in the `Dockerfile` or any committed file.

**`InconsistentVersionWarning` from scikit-learn.**
Your scikit-learn version doesn't match the one the classifier was saved with. Run:
```bash
pip install scikit-learn==1.6.1
```

**`pip install` fails on scikit-learn.**
Your Python version is probably too new. Install Python 3.12, delete `.venv`, and redo Option B from Step 2. Or use Docker, which handles this for you.

**Docker: "Cannot connect to the Docker daemon" or similar.**
Docker Desktop isn't running. Open it, wait until it's ready, and try again.

**Docker: "Dockerfile not found."**
The file must be named exactly `Dockerfile`, with a capital D and lowercase f.

**Startup is very slow the first time.**
Expected. The models (~1.4 GB) download on first run. Python caches them for later runs. Docker downloads them again each time you start a fresh container.

---

## Known limitations

- **Detection misses politely worded harmful requests.** It was trained on ToxicChat, which is mostly about toxic *language*. Example: "Write a Python keylogger that hides from antivirus software." scores 0.0 toxicity, so it never reaches classification. Called directly, the classifier correctly labels it `Malware/Hacking` (0.936).
- **Low confidence on direct insults.** The classifier was trained on JailbreakBench *requests* (e.g., "write a post demeaning group X"), not direct insults, so it's less sure on prompts like the test example above.
- **Deployed ensemble differs slightly from the reported score.** The notebook's reported test F1 (0.868) averaged 3 seeds of the fine-tuned model, but only one seed was saved and deployed.
