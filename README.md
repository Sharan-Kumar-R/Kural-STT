<h1 align="center">Kural-STT: SraVaani Speech-to-Text API for Indian-Language Calls</h1>
<br>
<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
</p>
<br>

🎧 Self-hosted speech-to-text for phone calls in Indian languages, served over a secure HTTP API.

Kural-STT runs [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) (ARTPARK / IISc, 65 Indian languages and dialects) on your own GPU or CPU. Upload a recording, or give its URL, and get the transcript back in the call's own script. No inference provider hosts SraVaani today, so this is how you run it as a service.

## Benchmark

200 real customer calls to an Indian gold-loan company, 40 each in Hindi, Kannada, Malayalam, Tamil and Telugu, 60 to 293 seconds long (8.4 hours of audio), October 2026.

An LLM judge (DeepSeek) read the English version of each call from every system with the names hidden and the order shuffled, rebuilt what was most likely said, and scored each version from 0 to 100 on the share of real content it captured: amounts, gold weight, names, the customer's request and the outcome. Each call was judged twice with a different order.

| System | Average score | Best of the three on | Badly transcribed calls (score < 40) |
| --- | ---: | ---: | ---: |
| Sarvam Saaras V3 (paid API) | **82.6** | 147 of 200 | 1 |
| **SraVaani-1.0 + LLM translation (this repo)** | **74.7** | 52 of 200 | 2 |
| Groq Whisper large-v3, translate mode | 55.5 | 1 of 200 | 30 |

**By language (average score):**

| | Hindi | Kannada | Malayalam | Tamil | Telugu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sarvam Saaras V3 | 84.2 | 83.3 | 81.2 | 83.2 | 80.9 |
| SraVaani + LLM translation | 81.6 | 71.8 | 73.0 | 70.7 | 76.5 |
| Groq Whisper large-v3 | 62.5 | 48.3 | 51.1 | 61.2 | 54.2 |

**Head to head:** SraVaani beats Groq Whisper by more than 5 points on 151 of 200 calls and loses on 24. Against Sarvam as the answer key, SraVaani captures 76.5% of the content against Groq's 57.6%, and loses a critical fact on 143 calls against 186.

**Speed:**

| System | Where | Time per call (average 2.5 min of audio) |
| --- | --- | ---: |
| Groq Whisper large-v3 | Groq cloud | 2.3 s |
| SraVaani speech-to-text | Laptop CPU, one worker | 20.3 s (7.5× real time) |
| SraVaani + LLM translation | Laptop CPU + Groq LLM | 26.2 s |

GPU speed has not been measured yet; see [Performance and sizing](#performance-and-sizing).

**Read these numbers with care:**
- Scores come from one LLM judge that cannot hear the audio. Ranking was stable across the two shuffled runs, but they are estimates, not human-verified accuracy.
- SraVaani writes the call's own script. The scores above use an LLM translation to English (Groq `openai/gpt-oss-120b`) so every system is judged in the same language. This API returns the native-script text only.
- The translation writes numbers as words ("one lakh"), which matters if your downstream steps expect digits.

## Features

- SraVaani-1.0 on GPU or CPU, with long calls split into 30-second pieces
- HTTP API: upload a file or give a recording URL, get the transcript and timing back
- Parallel model workers (8 by default), warmed up at startup and capped automatically to fit RAM and GPU memory
- API key required, private-network fetches blocked, size, time and queue limits
- Crash recovery: a worker that dies in native code is replaced without restarting the service
- Nothing stored: audio is deleted after each request, transcripts are not logged
- `check_setup.py` checks a new machine before you deploy
- systemd unit and HTTPS exposure through Cloudflare Tunnel

## How It Works

```
 client ──HTTPS──▶ Cloudflare Tunnel ──▶ API (FastAPI, one process, 127.0.0.1:8000)
                                           │  checks key, size, queue
                                           │  saves audio to a temp file
                                           ▼
                                  worker pool (8 processes)
                                  each holds one SraVaani copy on GPU/CPU
                                           │  16 kHz mono → 30 s pieces → text
                                           ▼
 client ◀──── JSON: text, audio length, timings ──── temp file deleted
```

## Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| Python | 3.10 | 3.10 |
| GPU | none (CPU works) | NVIDIA with 16 GB+ VRAM; a 32 GB V100 fits 8 workers |
| RAM | 8 GB for 1 worker | about 4.5 GB per CPU worker; 64 GB for 8 |
| CPU | 4 cores | 16 cores |
| Disk | 2 GB (model weights are 909 MB) | 5 GB |

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sharan-Kumar-R/Kural-STT.git
cd Kural-STT
```

### Step 2: Create and Activate a Virtual Environment

```bash
python3.10 -m venv venv
```

**For macOS/Linux:**
```bash
source venv/bin/activate
```

**For Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install PyTorch

Install PyTorch before the other dependencies, picking the build that matches your machine.

**NVIDIA Volta GPU (Tesla V100, Titan V):**
```bash
pip install torch==2.14.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
```
Volta works only with the CUDA 12.6 build. The default `pip install torch` gives a CUDA 13 build that has no V100 kernels and fails with `no kernel image is available`. PyTorch 2.14 is the last release with Volta support, so do not upgrade past it. The NVIDIA driver must be 560 or newer.

**NVIDIA Turing or newer (T4, RTX 20xx and later, A10, L4, A100, H100):**
```bash
pip install torch==2.14.0 torchaudio==2.11.0
```

**CPU only:**
```bash
pip install torch==2.14.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

MP3 decoding goes through `soundfile` (libsndfile), bundled with the wheel, so no separate FFmpeg install is needed.

## Configuration

### 1. Hugging Face Token

The SraVaani weights are gated.

1. Visit the [SraVaani-1.0 model page](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0)
2. Sign in and accept the access conditions
3. Open [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Create a token with **Read** access

The first start downloads about 909 MB of weights into `hf/`.

### 2. Environment File

```bash
cp .env.example .env
```

Set at least `HF_TOKEN` and `SRAVAANI_API_KEY`. Generate a key with:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | Yes | Downloads the gated SraVaani weights |
| `SRAVAANI_API_KEY` | Yes | Key every request must send; at least 24 random characters |
| `SRAVAANI_HOST` / `SRAVAANI_PORT` | No | Where the API listens; default `127.0.0.1` / `8000`. Keep `127.0.0.1` when exposing through a tunnel |
| `SRAVAANI_MAX_UPLOAD_MB` | No | Largest recording accepted; default `100` (a 10-minute telephony MP3 is about 1.2 MB) |
| `SRAVAANI_REQUEST_TIMEOUT` | No | Seconds before giving up on one recording; default `1800` |
| `SRAVAANI_MAX_QUEUE` | No | Requests allowed to wait beyond the busy workers before new ones get `503`; default `32` |
| `SRAVAANI_ALLOW_URL` | No | `0` disables the `url` option so only uploads are accepted; default `1` |
| `SRAVAANI_ALLOW_PRIVATE_URLS` | No | `1` lets `url` reach private or internal addresses; default `0`, keep it that way on an exposed server |
| `SRAVAANI_ALLOW_NO_KEY` | No | `1` starts without a key; only on a fully private network; default `0` |
| `SRAVAANI_MODEL` | No | Hugging Face model id; default `ARTPARK-IISc/SraVaani-1.0` |
| `SRAVAANI_WORKERS` | No | Model copies serving requests in parallel; default `8` |
| `SRAVAANI_DEVICE` | No | `auto` (GPU if visible, else CPU), `cuda`, `cuda:1` or `cpu`; default `auto` |
| `SRAVAANI_THREADS` | No | CPU threads per worker; default splits the cores evenly on CPU, `2` on GPU |
| `SRAVAANI_PIECE_BATCH` | No | 30-second pieces sent to the model together; default `8`. Lower it to save memory |
| `SRAVAANI_WORKER_RAM_GB` | No | RAM budget per worker, used to cap the worker count; default `4.5` |
| `SRAVAANI_WORKER_GPU_GB` | No | GPU memory budget per worker, used to cap the worker count; default `3.5` |
| `SRAVAANI_FP16` | No | `1` runs the model in half precision on GPU: about half the VRAM, accuracy not yet checked; default `0` |
| `SRAVAANI_JIT_OPT` | No | `1` keeps TorchScript graph optimisation on. Default off on Windows, where it crashes natively, and on elsewhere |
| `SRAVAANI_TMP_DIR` | No | Temp folder, useful when the system drive is short on space |
| `SRAVAANI_LOG_DIR` | No | Where native crash traces are written; default `logs/` |

**Important:** `.env` is gitignored and must never be committed.

## Check the Machine

```bash
python check_setup.py
```

It checks Python, the PyTorch build and whether it supports your GPU, cores, RAM, GPU memory and the token, then times one worker on generated audio. It needs no recordings. Fix any `FAIL` line before going further.

## Run the API

```bash
python -m kural_stt.server
```

It loads every worker, prints `ready`, and listens on `SRAVAANI_HOST:SRAVAANI_PORT`. Run it as a single process; the parallelism is in its worker pool. Interactive docs are at `/docs`.

## Using the API

**Upload a file:**
```bash
curl -H "X-API-Key: $KEY" -F "file=@call.mp3" -F "call_id=abc123" https://stt.example.com/transcribe
```

**Or give a recording URL** (with an optional `Referer`, which some telephony providers require):
```bash
curl -H "X-API-Key: $KEY" -F "url=https://example.com/recording.mp3" -F "referer=https://example.com/" https://stt.example.com/transcribe
```

**Python:**
```python
import requests

with open("call.mp3", "rb") as f:
    r = requests.post("https://stt.example.com/transcribe", headers={"X-API-Key": KEY},
                      files={"file": f}, data={"call_id": "abc123"}, timeout=1800)
r.raise_for_status()
print(r.json()["text"])
```

**Response:**
```json
{
 "call_id": "abc123",
 "text": "native-script transcript ...",
 "audio_sec": 96.2,
 "infer_sec": 13.6,
 "pieces": 4,
 "device": "cuda",
 "bytes": 192384,
 "total_sec": 13.77,
 "x_real_time": 7.1
}
```

`infer_sec` is model time; `total_sec` also includes the upload and any wait in the queue. When more requests arrive than there are workers, the extra ones wait their turn.

| Endpoint | Purpose |
| --- | --- |
| `POST /transcribe` | One recording in (`file` upload or `url` form field, optional `referer` and `call_id`), transcript out |
| `GET /health` | Device, worker count, busy workers, requests served and failed; no key needed |
| `GET /docs` | Interactive API documentation |

| Status | Meaning |
| --- | --- |
| `400` | Sent both or neither of `file` and `url`, or a `url` that is not allowed |
| `401` | Missing or wrong API key |
| `413` | Larger than `SRAVAANI_MAX_UPLOAD_MB` |
| `415` | Not a readable audio file |
| `502` | The `url` could not be downloaded |
| `503` | Queue full; retry after the `Retry-After` seconds |
| `504` | Took longer than `SRAVAANI_REQUEST_TIMEOUT` |

Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` and anything else libsndfile or audioread can decode. Telephony audio at 8 kHz is fine; it is resampled to 16 kHz.

## Security

- **API key required.** The server refuses to start without `SRAVAANI_API_KEY`, and rejects keys shorter than 24 characters or still set to the example value. Keys are compared in constant time. Send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- **No internal fetches.** The `url` option only downloads from public internet addresses, checked again on every redirect, so it cannot be used to reach the host's own network, router or cloud metadata. Turn it off entirely with `SRAVAANI_ALLOW_URL=0`.
- **Size, time and queue limits.** Oversized uploads are rejected while streaming; slow recordings time out; a full queue answers `503`.
- **Nothing kept.** Audio is deleted after each request; transcripts are not stored or logged.
- **HTTPS is your job.** The API speaks plain HTTP on `127.0.0.1`. Put HTTPS in front of it (below) before real calls travel over the internet, and never expose port 8000 directly.

## Deployment

On the GPU server (Linux), after installation and configuration:

**1. Check the machine:** `python check_setup.py`

**2. Run it as a service** so it starts on boot and restarts after a crash. The unit file assumes the repo is at `/opt/Kural-STT` and runs as a user named `kural`; edit both to match:

```bash
sudo cp deploy/kural-stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kural-stt
journalctl -u kural-stt -f          # wait for "ready"
curl http://127.0.0.1:8000/health
```

**3. Expose it over HTTPS with Cloudflare Tunnel.** No router port forwarding, no public IP and no certificate setup; the home or office network stays closed.

Quick test (temporary random `https://...trycloudflare.com` address that changes on every restart):
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Permanent address on your own domain (free Cloudflare account with the domain added):
```bash
cloudflared tunnel login
cloudflared tunnel create kural-stt
cloudflared tunnel route dns kural-stt stt.example.com
cloudflared tunnel run --url http://127.0.0.1:8000 kural-stt
```
Then run `sudo cloudflared service install` so the tunnel survives reboots too.

**4. Hand over two things:** the HTTPS address (for example `https://stt.example.com`) and the API key, sent separately and privately. Callers use `https://stt.example.com/transcribe` as in the examples above.

**Alternatives:** Tailscale (a private network between the two machines, nothing public at all), or a reverse proxy such as Caddy in front of an open port, which adds HTTPS automatically.

## Performance and Sizing

| Measured | Value |
| --- | --- |
| Model download | 909 MB (about 450 million parameters, stored in half precision) |
| CPU RAM per worker, peak | 4.3 GB (8 pieces of 30 s per batch) |
| Laptop CPU, 1 worker | 7.5× real time |
| Laptop CPU, 2 workers | 7.7× real time overall |

| Estimated, not yet measured | FP32 (default) | FP16 (`SRAVAANI_FP16=1`) |
| --- | ---: | ---: |
| VRAM per worker | about 2.5 to 3.5 GB | about 1.5 to 2 GB |
| VRAM for 8 workers | about 20 to 28 GB | about 12 to 16 GB |

Call length barely changes memory, because every call is cut into 30-second pieces; a 50-minute call peaks at about the same memory as a 2-minute one and just takes longer. Measure real VRAM while the API is busy with `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2`, then set `SRAVAANI_WORKER_GPU_GB` to match.

## Troubleshooting

- **`Set HF_TOKEN in .env`**: the token is missing, or you have not accepted SraVaani's access conditions on Hugging Face
- **`Set SRAVAANI_API_KEY`** / **`SRAVAANI_API_KEY is too weak`** at startup: generate a key as shown in Configuration
- **`no kernel image is available for execution on the device`**: the PyTorch build does not support your GPU. For a V100, reinstall from the `cu126` index (Installation, Step 3)
- **`capping workers at N`** in the log: RAM or GPU memory cannot hold `SRAVAANI_WORKERS` copies, so fewer were started. Add memory, set `SRAVAANI_FP16=1`, or lower `SRAVAANI_PIECE_BATCH`
- **`not enough memory` / `CUDA out of memory`**: lower `SRAVAANI_WORKERS` or `SRAVAANI_PIECE_BATCH`, or raise `SRAVAANI_WORKER_RAM_GB` / `SRAVAANI_WORKER_GPU_GB`
- **`worker crashed natively; restarting the pool`** in the log: the model's native code crashed; the pool is rebuilt automatically. If it repeats on Linux, set `SRAVAANI_JIT_OPT=0`. Traces are in `logs/fault.log`
- **GPU gets slower over time, or the machine shuts down**: a passively cooled data-centre card (such as a PCIe V100) needs forced airflow. Watch it with `nvidia-smi -q -d TEMPERATURE,PERFORMANCE`
- **`502` on a `url`**: the recording host refused the download; many telephony providers need a `referer`, or links expire
- **Model download fails or the system drive fills up**: set `SRAVAANI_TMP_DIR` to a folder on a larger drive

## Project Structure

```
Kural-STT/
├── .env.example                # Template for .env
├── .gitattributes              # Keeps shell files on LF line endings
├── .gitignore                  # Keeps keys, audio, logs and model weights out of git
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
├── check_setup.py              # Checks a new machine before deployment
├── deploy/
│   └── kural-stt.service       # systemd unit for running the API permanently
└── kural_stt/
    ├── __init__.py
    ├── settings.py             # Settings read from .env
    ├── transcribe.py           # SraVaani engine: device, worker sizing, model loading, transcription
    └── server.py               # HTTP API, worker pool and security checks
```

## Privacy

Call recordings contain real people's voices, names and amounts. Only process audio you are allowed to process, keep the API behind HTTPS and a strong key, and never commit `.env`, recordings or transcripts. The service deletes audio after each request and does not store transcripts.

## Acknowledgements

- [SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) by ARTPARK and IISc (MIT licence)
- Benchmark references: [Sarvam Saaras V3](https://www.sarvam.ai/) and [Whisper large-v3](https://github.com/openai/whisper) served by [Groq](https://groq.com/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

In case of any queries, please leave a message or contact me via the email provided in my profile.

<p align="center">
⭐ <strong>Star this repository if you found it helpful!</strong>
</p>
