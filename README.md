<h1 align="center">SraVaani vs Groq Whisper: Indian-Language Call Transcription Benchmark</h1>
<br>
<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/groq-FF6600?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
</p>
<br>

🎧 Real phone calls in Tamil, Kannada and Telugu, transcribed two ways and judged by the same LLM.

This project compares two routes from call audio to an English transcript:

   ⚡ **Groq Whisper large-v3 (translate)**: speech straight to English in one API call

   🇮🇳 **SraVaani-1.0 + LLM translation**: [ARTPARK / IISc's SraVaani](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) writes native-script text on your own GPU or CPU, then a Groq-hosted LLM translates it to English

Both English transcripts are judged by the same Groq LLM prompt, so the transcript is the only thing that differs between the two sides. The benchmark counts how often each route leaves a real customer request hidden as "NOISE", how often the two agree on the caller's intent, and how fast SraVaani runs on your own hardware.

## Features

- Works on any folder of call recordings: no database or private backend needed
- Reproducible sample: a fixed seed picks the same calls every time
- SraVaani inference in parallel worker processes (8 by default), on GPU when one is present, in 30-second pieces, with crash-safe resume
- Worker count capped automatically to fit RAM and GPU memory
- `check_setup.py` checks a new machine (PyTorch build, GPU, cores, RAM, token) before you run real calls
- Groq Whisper baseline with automatic retry on rate limits
- Faithful-translation prompt and a NOISE / intent judge, both on Groq
- Optional fixed intent list, so both sides are scored on your own categories
- Plain-text reports: per-group counts, disagreements, intent differences and side-by-side examples

## How It Works

```
 data/audio/<group>/*.mp3  +  eval_config.json
        │
        ▼
 1. sample      ──▶ data/manifest.json
        │
        ├──────────────────────────────┐
        ▼                              ▼
 2. transcribe (SraVaani,        3. baseline (Groq Whisper)
    GPU or CPU, 8 workers)
    data/sravaani_out.jsonl         data/groq_out.jsonl
        │                              │
        └──────────────┬───────────────┘
                       ▼
 4. compare     ──▶ data/compare_out.jsonl
        │             (SraVaani → English, then both judged by one prompt)
        ▼
 5. report      ──▶ outputs/analysis.txt + outputs/examples.txt
```

Every step appends one JSON line per call and skips calls it has already done, so any step can be stopped and re-run safely. Steps 2 and 3 are independent and can run at the same time.

## Installation

### Step 1: Clone the Repository

**Option A: Using VS Code Terminal**
1. Open Visual Studio Code
2. Open a new terminal (Terminal → New Terminal or ``Ctrl+Shift+` ``)
3. Navigate to your desired directory:
   ```bash
   cd path/to/your/desired/folder
   ```
4. Clone the repository:
   ```bash
   git clone https://github.com/Sharan-Kumar-R/Kural-STT.git
   ```
5. Open the project folder:
   ```bash
   cd Kural-STT
   ```
6. Open the project in VS Code:
   ```bash
   code .
   ```

**Option B: Using VS Code Git Integration**
1. Open Visual Studio Code
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Type "Git: Clone" and select it
4. Paste the repository URL: `https://github.com/Sharan-Kumar-R/Kural-STT.git`
5. Choose a folder location and click "Select Repository Location"
6. Click "Open" when prompted

### Step 2: Check Python

Python **3.10** is required. Verify it with:

```bash
python --version
```

### Step 3: Create Virtual Environment

Open a terminal in Visual Studio Code and run:

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

**For Windows:**
```bash
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
source venv/bin/activate
```

After activation, you should see something like this in your terminal:
```
(venv) PS C:\Users\username\path\to\Kural-STT>
```

### Step 5: Install PyTorch

Install PyTorch before the other dependencies, picking the build that matches your machine.

**CPU only:**
```bash
pip install torch==2.14.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA Volta GPU (Tesla V100, Titan V):**
```bash
pip install torch==2.14.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
```
Volta works only with the CUDA 12.6 build. The default `pip install torch` gives a CUDA 13 build that has no V100 kernels and fails with `no kernel image is available`. PyTorch 2.14 is the last release with Volta support, so do not upgrade past it. The NVIDIA driver must be 560 or newer.

**NVIDIA Turing or newer (T4, RTX 20xx and later, A10, L4, A100, H100):**
```bash
pip install torch==2.14.0 torchaudio==2.11.0
```

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

MP3 decoding goes through `soundfile` (libsndfile), which is bundled with the wheel, so no separate FFmpeg install is needed.

## API Setup

### 1. Hugging Face Token (SraVaani access)

The SraVaani weights are gated.

1. Visit the [SraVaani-1.0 model page](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0)
2. Sign in and accept the access conditions
3. Open [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Create a token with **Read** access
5. Copy the token for later use

The first run downloads about 870 MB of weights into `hf/`.

### 2. Groq API Key (baseline, translation and judging)

1. Visit [Groq Console](https://console.groq.com/)
2. Create an account or sign in
3. Open **API Keys** and create a new key
4. Copy the API key for later use

Groq is used three times per call: Whisper for the baseline transcript, and the LLM for translating SraVaani's text and for judging both English transcripts.

### 3. Environment Configuration

Copy the example file and fill in your keys:

**For Windows:**
```bash
copy .env.example .env
```

**For macOS/Linux:**
```bash
cp .env.example .env
```

```env
HF_TOKEN=your_hugging_face_token_here
GROQ_API_KEY=your_groq_api_key_here

GROQ_STT_MODEL=whisper-large-v3
GROQ_LLM_MODEL=openai/gpt-oss-120b
GROQ_WORKERS=4
GROQ_BASE_URL=https://api.groq.com/openai/v1

SRAVAANI_MODEL=ARTPARK-IISc/SraVaani-1.0
SRAVAANI_WORKERS=8
SRAVAANI_DEVICE=auto
SRAVAANI_THREADS=
SRAVAANI_PIECE_BATCH=8
SRAVAANI_WORKER_RAM_GB=4.5
SRAVAANI_WORKER_GPU_GB=3.5
SRAVAANI_FP16=0
SRAVAANI_JIT_OPT=
SRAVAANI_TMP_DIR=
SRAVAANI_CONFIG=
SRAVAANI_DATA_DIR=
SRAVAANI_OUTPUT_DIR=
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | Yes | Downloads the gated SraVaani weights |
| `GROQ_API_KEY` | Yes | Groq Whisper baseline, translation and judging |
| `GROQ_STT_MODEL` | No | Groq speech model for the baseline; default `whisper-large-v3` |
| `GROQ_LLM_MODEL` | No | Groq chat model for translating and judging; default `openai/gpt-oss-120b` |
| `GROQ_WORKERS` | No | Parallel Groq requests; default `4`. Lower it if you hit rate limits |
| `GROQ_BASE_URL` | No | Any OpenAI-compatible endpoint; default is Groq's |
| `SRAVAANI_MODEL` | No | Hugging Face model id; default `ARTPARK-IISc/SraVaani-1.0` |
| `SRAVAANI_WORKERS` | No | Parallel transcription processes, each with its own model copy; default `8` |
| `SRAVAANI_DEVICE` | No | `auto` (GPU if visible, else CPU), `cuda`, `cuda:1` or `cpu`; default `auto` |
| `SRAVAANI_THREADS` | No | CPU threads per worker; default splits the cores evenly on CPU, `2` on GPU |
| `SRAVAANI_PIECE_BATCH` | No | 30-second pieces sent to the model together; default `8`. Lower it to save memory |
| `SRAVAANI_WORKER_RAM_GB` | No | RAM budget per worker, used to cap the worker count; default `4.5` (measured peak about 4.3 GB on CPU) |
| `SRAVAANI_WORKER_GPU_GB` | No | GPU memory budget per worker, used to cap the worker count; default `3.5` (an estimate; check with `nvidia-smi`) |
| `SRAVAANI_FP16` | No | `1` runs the model in half precision on GPU: faster, accuracy not yet checked; default `0` |
| `SRAVAANI_JIT_OPT` | No | `1` keeps TorchScript graph optimisation on. Default off on Windows, where it crashes natively, and on elsewhere |
| `SRAVAANI_TMP_DIR` | No | Temp folder for model loading, useful when the system drive is short on space |
| `SRAVAANI_CONFIG` | No | Path to the config file; default `eval_config.json` |
| `SRAVAANI_DATA_DIR` | No | Where audio and per-step results live; default `data/` |
| `SRAVAANI_OUTPUT_DIR` | No | Where reports are written; default `outputs/` |

**Important:** Replace the placeholder values with your own. `.env` is gitignored and must never be committed.

### 4. Evaluation Configuration

Copy the example config:

**For Windows:**
```bash
copy eval_config.example.json eval_config.json
```

**For macOS/Linux:**
```bash
cp eval_config.example.json eval_config.json
```

```json
{
 "seed": 42,
 "call_context": "a call to an Indian gold-loan company",
 "intents": ["Gold Rate & Loan Amount", "Interest Rate & Charges", "Loan Renewal & Top-Up", "Other Request"],
 "groups": [
  {"label": "Tamil Nadu", "language": "Tamil", "audio_subdir": "tamil", "calls": 20},
  {"label": "Karnataka", "language": "Kannada", "audio_subdir": "kannada", "calls": 20}
 ],
 "examples": []
}
```

| Key | Meaning |
| --- | --- |
| `seed` | Random seed, so the same calls are picked every time |
| `call_context` | Fills both prompts: "…from `<call_context>`" |
| `intents` | Allowed intents for the judge. Leave `[]` to let it write a short free label |
| `groups[].label` | Group name shown in reports |
| `groups[].language` | Spoken language, passed to the translation prompt |
| `groups[].audio_subdir` | Folder under `data/audio/` holding this group's recordings |
| `groups[].calls` | Calls to pick at random; leave it out to use every file |
| `groups[].client` | Optional tag, shown in reports when you compare several businesses |
| `groups[].intents` | Optional intent list that overrides the top-level one for this group |
| `examples` | Call-id prefixes (file names) to print side by side in `outputs/examples.txt` |

### 5. Add Your Recordings

Put recordings in one folder per group. The file name (without extension) becomes the call id:

```
data/audio/
├── tamil/
│   ├── call_0001.mp3
│   └── call_0002.mp3
└── kannada/
    └── call_0101.mp3
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`. Telephony audio at 8 kHz is fine; it is resampled to 16 kHz.

## Usage

Make sure your virtual environment is activated:

```bash
venv\Scripts\activate
```

Run each step from the project root, in order.

### Step 1: Build the Sample

```bash
python -m sravaani_eval.sample
```

Lists the recordings in each group, picks `calls` of them at random and writes `data/manifest.json`:

```
Tamil Nadu: 64 recordings -> 20 chosen
Karnataka: 41 recordings -> 20 chosen
manifest: 40 calls
```

### Step 2: Transcribe with SraVaani

On a new machine, check the setup first. It needs no recordings: it checks the PyTorch build, GPU, cores, RAM and token, then times a short transcription of generated audio:

```bash
python check_setup.py
```

Then transcribe:

```bash
python -m sravaani_eval.transcribe
```

Each call is resampled to 16 kHz mono and split into 30-second pieces. `SRAVAANI_WORKERS` processes (8 by default) each load their own model copy and take calls from a shared queue, on the GPU when one is visible. The worker count is lowered automatically when RAM or GPU memory cannot hold that many copies, and the log says so. Progress lines show the running speed as a multiple of real time.

A worker can die in native code (seen on Windows). The calls it was holding are then retried one at a time, and a call that crashes twice is recorded with an error instead of blocking the run. Calls that failed with an ordinary error are retried on the next run. The wrapper scripts restart the step until every call is done:

```bash
./run_transcribe.sh          # Linux / macOS
```
```powershell
.\run_transcribe.ps1         # Windows
```

Measured speeds:

| Machine | Workers | Speed |
| --- | --- | --- |
| Laptop, 16 threads, 17 GB RAM, CPU | 1 | 7.4× real time |
| Same laptop | 2 (capped from 8 by RAM) | 7.7× real time |

A 16-core desktop with 64 GB RAM can run all 8 workers on CPU. GPU speed has not been measured yet: run `check_setup.py`, then a real batch, and add your numbers here.

### Step 3: Run the Groq Whisper Baseline

```bash
python -m sravaani_eval.baseline
```

Sends each recording to Groq's Whisper translation endpoint and writes its English text to `data/groq_out.jsonl`. Groq accepts files up to 25 MB.

### Step 4: Translate and Judge

```bash
python -m sravaani_eval.compare
```

For each call that has both transcripts it:
1. Translates SraVaani's native-script text to English with the Groq LLM (temperature 0)
2. Judges Groq Whisper's English: NOISE or real, intent, one-line summary
3. Judges SraVaani's English with the same prompt
4. Writes both verdicts to `data/compare_out.jsonl`

Transcripts under three words are marked NOISE without calling the LLM.

### Step 5: Write the Reports

```bash
python -m sravaani_eval.report
```

Writes:
- `outputs/analysis.txt`: per-group NOISE counts, intent agreement, errors, speed and every disagreement
- `outputs/examples.txt`: Groq English, SraVaani native text and SraVaani English for the configured example calls

To print other examples, pass call-id prefixes:

```bash
python -m sravaani_eval.report call_0001 call_0101
```

### Reading the Report

| Column | Meaning |
| --- | --- |
| `groq NOISE` / `sravaani NOISE` | Calls the judge marked NOISE on each side |
| `both real` | Calls where both sides found a real request |
| `same intent (both real)` | Of those, how many got the same intent (sub-types such as `- GL` ignored) |
| `groq->NOISE only` | Groq NOISE, SraVaani real: a request Groq hid |
| `sravaani->NOISE only` | SraVaani NOISE, Groq real: a request SraVaani lost |
| `errors g/s` | Judge errors on the Groq / SraVaani side |

Set `intents` in the config before trusting the `same intent` column: with free labels, "gold loan inquiry" and "loan enquiry" count as different.

## Example Results

From the first run: 90 real calls, 2.96 h of audio, September 2026. That run scored both sides with a production call-analysis pipeline instead of this repo's judge prompt, so your numbers will differ.

| Group | Calls | NOISE with Groq | NOISE with SraVaani | Groq NOISE, SraVaani real | SraVaani NOISE, Groq real |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gold loans · Tamil | 20 | 9 | 4 | 5 | 0 |
| Gold loans · Kannada | 20 | 9 | 6 | 5 | 2 |
| Gold loans · Telugu | 20 | 6 | 4 | 3 | 1 |
| Vehicle finance · Telugu | 30 | 15 | 15 | 0 | 0 |
| **All** | **90** | **39** | **29** | **13** | **3** |

On the gold-loan calls, SraVaani plus translation cut NOISE from 40% to 23%, mostly short calls where Whisper's direct translation produced a few unrelated English words. The vehicle-finance calls did not change: their NOISE is IVR-menu and hold-queue audio that no transcription model can fix.

**Limits:** 20 to 30 calls per group, no human listening yet, and the result depends on the translation prompt as much as on SraVaani.

## Deactivating the Environment

When you're done working with the project, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

- **`Set HF_TOKEN in .env`**: the token is missing, or you have not accepted SraVaani's access conditions on Hugging Face
- **`Set GROQ_API_KEY in .env`**: the Groq key is missing
- **`Copy eval_config.example.json to eval_config.json`**: the config is missing
- **`Missing audio folder …`**: a group's `audio_subdir` does not exist under `data/audio/`
- **`HTTPError: 429`** after retries: lower `GROQ_WORKERS`, or wait for your Groq rate limit to reset
- **`HTTPError: 413`** in the baseline: the recording is over Groq's 25 MB limit; convert it to mono MP3 first
- **`HTTPError: 404`** on a model: that model was retired on Groq; pick a current one from the [Groq models page](https://console.groq.com/docs/models) and set `GROQ_STT_MODEL` or `GROQ_LLM_MODEL`
- **Transcribe exits with no error partway through**: a native crash; run `./run_transcribe.sh` or `.\run_transcribe.ps1`, which restarts it and resumes
- **`no kernel image is available for execution on the device`**: the PyTorch build does not support your GPU. For a V100, reinstall from the `cu126` index (Installation, Step 5)
- **`not enough memory` / `CUDA out of memory`**: lower `SRAVAANI_WORKERS` or `SRAVAANI_PIECE_BATCH`, or raise `SRAVAANI_WORKER_RAM_GB` / `SRAVAANI_WORKER_GPU_GB` so fewer workers start
- **GPU gets slower over time, or the machine shuts down**: a passively cooled data-centre card (such as a PCIe V100) needs forced airflow. Watch it with `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` while it runs
- **Model download fails or the system drive fills up**: set `SRAVAANI_TMP_DIR` to a folder on a larger drive
- **A step does nothing**: every call is already done; delete that step's output file in `data/` to redo it
- Check installed packages with `pip list`, and make sure the virtual environment is active

## Project Structure

```
Kural-STT/
├── .env                        # Your keys (gitignored)
├── .env.example                # Template for .env
├── .gitignore                  # Keeps keys, audio, data and model weights out of git
├── eval_config.json            # Your groups and intents (gitignored)
├── eval_config.example.json    # Template for eval_config.json
├── README.md                   # This documentation
├── check_setup.py              # Checks a new machine before real runs
├── requirements.txt            # Python dependencies
├── run_transcribe.sh           # Linux/macOS: restarts the transcribe step until every call is done
├── run_transcribe.ps1          # Windows: restarts the transcribe step until every call is done
├── sravaani_eval/
│   ├── __init__.py
│   ├── settings.py             # Paths, model ids, API settings and config loading
│   ├── store.py                # JSON and JSON Lines helpers
│   ├── groq_client.py          # Groq Whisper and chat calls with retry
│   ├── sample.py               # Step 1: pick recordings into a manifest
│   ├── transcribe.py           # Step 2: SraVaani transcription, parallel workers, GPU or CPU
│   ├── baseline.py             # Step 3: Groq Whisper speech-to-English
│   ├── compare.py              # Step 4: translate and judge both routes
│   └── report.py               # Step 5: analysis and example reports
├── data/                       # Audio, manifest and per-step results (gitignored)
├── outputs/                    # Generated reports (gitignored)
├── hf/                         # Hugging Face model cache (gitignored)
└── venv/                       # Virtual environment (gitignored)
```

## Privacy

Call recordings, transcripts and reports contain real people's voices, names and amounts. `data/`, `outputs/`, every audio format, `.env` and `eval_config.json` are gitignored. Keep them that way, and never paste transcript text into issues or pull requests. Sending recordings to Groq sends them to a third-party API: make sure you are allowed to.

## Acknowledgements

- [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) by ARTPARK and IISc (MIT licence)
- [Whisper large-v3](https://github.com/openai/whisper) by OpenAI, served by [Groq](https://groq.com/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

In case of any queries, please leave a message or contact me via the email provided in my profile.

<p align="center">
⭐ <strong>Star this repository if you found it helpful!</strong>
</p>
