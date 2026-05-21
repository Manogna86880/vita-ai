# Vita AI

> **AI-powered preliminary health screening prototype.**
> Analyses face video, breathing audio, and symptom text to produce a combined Vita Health Score with risk labels, confidence values, and human-readable recommendations.

---

## ⚠️ Disclaimer

**Vita AI is a hackathon prototype for preliminary health screening only.**
It does **not** provide medical diagnoses.  Always consult a qualified healthcare professional for clinical advice.

---
## 👥 Team Members
This project was designed and developed as a collaborative team effort for the hackathon by:
* **Poludasu Manogna**  — [@Manogna86880](https://github.com/Manogna86880)
* **Aryan** — [@aryangopisetty9](https://github.com/aryangopisetty9)
* Poludasu Akshaya
* Sreekar mani

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Face Video  │   │ Breathing    │   │  Symptom     │
│  (rPPG)      │   │  Audio       │   │  Text        │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │ face_module │   │audio_module │   │symptom_mod. │
 │ (OpenCV +   │   │ (Librosa +  │   │ (HF transformers
 │  MediaPipe  │   │  signal +   │   │  + BioBERT  │
 │  + Open-rPPG│   │  YAMNet)    │   │  + keywords)│
 │  primary +  │   │             │   │             │
 │  classical  │   │             │   │             │
 │  fallback)  │   │             │   │             │
 └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
        │                 │                  │
        └────────┬────────┴──────────────────┘
                 ▼
      ┌──────────────────┐
      │  Clinical        │
      │  Validation      │
      │  (consistency,   │
      │   escalation)    │
      └────────┬─────────┘
               ▼
        ┌──────────────┐
        │ Score Engine  │
        │ (trained      │
        │  fusion /     │
        │  weighted sum)│
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │  FastAPI      │
        │  REST + WS    │
        │  + Middleware  │
        └──────────────┘
```

All core logic lives in `models/`.  The API layer is a thin wrapper that exposes modules via HTTP.

---

## Folder Structure

```
vita-ai/
 ├── api/
 │    ├── main.py              # FastAPI app with all endpoints
 │    ├── middleware.py         # Request logging & exception handlers
 │    └── streaming.py         # WebSocket audio streaming router
 ├── models/
 │    ├── face_module.py       # Heart-rate estimation (rPPG)
 │    ├── audio_module.py      # Breathing analysis
 │    ├── symptom_module.py    # Symptom risk analysis
 │    ├── score_engine.py      # Combined Vita Health Score
 │    ├── model_registry.py    # Central pretrained model registry
 │    ├── fusion_model.py      # Trained ML fusion model loader
 │    ├── open_rppg_backend.py  # Primary rPPG backend (Open-rPPG package)
 │    ├── rppg_models.py       # rPPG model chain (Open-rPPG)
 │    ├── audio_models.py      # YAMNet
 │    └── nlp_models.py        # BioBERT / DistilBERT
 ├── core/
 │    ├── clinical_validation.py # Cross-module consistency & safety
 │    ├── validation.py        # Upload file & text validation
 │    ├── exceptions.py        # Typed exception hierarchy
 │    ├── logging_config.py    # Structured JSON logging
 │    ├── model_paths.py       # Cache directory & path management
 │    ├── model_download.py    # Auto-download logic for supported models
 │    └── model_status.py      # Runtime model status tracking
 ├── services/
 │    └── session_manager.py   # WebSocket stream session tracker
 ├── train/
 │    └── train_fusion_model.py # Fusion model training script
 ├── eval/
 │    ├── eval_rppg.py         # rPPG benchmark evaluation
 │    ├── eval_audio.py        # Audio module evaluation
 │    ├── eval_symptom.py      # Symptom module evaluation
 │    ├── run_all_evals.py     # Run all evaluations
 │    ├── metrics.py           # Regression & classification metrics
 │    └── results/             # Evaluation output files
 ├── tests/
 │    ├── test_face.py
 │    ├── test_audio.py
 │    ├── test_symptom.py
 │    ├── test_score_engine.py
 │    ├── test_pretrained_models.py
 │    ├── test_new_features.py # Registry, fusion, validation tests
 │    ├── test_open_rppg_backend.py # Open-rPPG backend wrapper tests
 │    ├── test_face_module_open_rppg.py # Face module Open-rPPG integration
 │    ├── test_health_status.py    # Health endpoint Open-rPPG status
 │    ├── test_face_endpoint_models.py # Model chain & backward compat
 │    ├── test_model_download.py # Download/cache system tests
 │    ├── test_model_status.py   # Model status tracking tests
 │    ├── test_nlp_model_loading.py  # NLP cache integration tests
 │    └── test_audio_model_loading.py # Audio cache integration tests
 ├── schemas.py                # Pydantic request / response models
 ├── config.py                 # Central configuration & constants
 ├── database.py               # SQLAlchemy DB setup
 ├── db_models.py              # ORM models (User, HealthData, ScanResult)
 ├── requirements.txt
 ├── .env.example
 └── README.md
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.11.9** (required)
- Windows, macOS, or Linux

### 2. Create and activate virtual environment

```bash
# Create venv with Python 3.11.9
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1
# or (Windows CMD)
venv\Scripts\activate.bat
# or (macOS / Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Open-rPPG (primary face AI model)

Open-rPPG is installed automatically with `requirements.txt`. To verify:

```bash
python scripts/setup_open_rppg.py              # Verify install + load default model
python scripts/setup_open_rppg.py --list        # List all 17 supported models
python scripts/setup_open_rppg.py --model PhysFormer.rlap  # Test a specific model
```

Open-rPPG ships with **pretrained weights bundled in the pip package** — no manual downloads needed.
Default model: `FacePhys.rlap`. Change with `VITA_OPEN_RPPG_MODEL=PhysFormer.rlap`.

### 5. Download other pretrained models

```bash
python scripts/download_supported_models.py   # DistilBERT, BioBERT, YAMNet
```

This will auto-download and cache:
- **DistilBERT** → `models_cache/distilbert/`
- **BioBERT** → `models_cache/biobert/`
- **YAMNet** → `models_cache/yamnet_saved_model/`

### 6. Run the API

```bash
uvicorn api.main:app --reload
```

The server starts at **http://127.0.0.1:8000**.  Interactive docs at http://127.0.0.1:8000/docs.

### 7. Run tests

```bash
pytest
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Project info & endpoint list |
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Readiness probe with model & stream status |
| `GET` | `/models` | List loaded pretrained models per module |
| `POST` | `/auth/signup` | Register a new user |
| `POST` | `/auth/login` | Authenticate user |
| `POST` | `/predict/face` | Upload face video → heart-rate estimate |
| `POST` | `/predict/audio` | Upload breathing audio → respiratory analysis |
| `POST` | `/predict/symptom` | Submit symptom text → risk classification |
| `POST` | `/predict/final-score` | Combine module outputs → Vita Health Score |
| `POST` | `/user/{id}/health-data` | Save user health profile |
| `GET` | `/user/{id}/health-data` | Retrieve user health profile |
| `POST` | `/user/{id}/scan` | Persist a scan result |
| `GET` | `/user/{id}/scans` | Get scan history |
| `WS` | `/ws/face-scan` | Real-time face scan via WebSocket |
| `WS` | `/ws/audio-stream` | Real-time audio streaming via WebSocket |

---

## Example `curl` Commands

### Symptom analysis

```bash
curl -X POST http://127.0.0.1:8000/predict/symptom \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel tired and dizzy with mild headache"}'
```

### Face video upload

```bash
curl -X POST http://127.0.0.1:8000/predict/face \
  -F "file=@sample_data/sample_face_video.mp4"
```

### Audio upload

```bash
curl -X POST http://127.0.0.1:8000/predict/audio \
  -F "file=@sample_data/sample_breathing.wav"
```

### Combined score

```bash
curl -X POST http://127.0.0.1:8000/predict/final-score \
  -H "Content-Type: application/json" \
  -d '{
    "face_result":    {"value": 74, "risk": "low", "confidence": 0.82},
    "audio_result":   {"value": 16, "risk": "low", "confidence": 0.76},
    "symptom_result": {"risk": "low", "confidence": 0.90}
  }'
```

---

## Module Details

### Face Module (`models/face_module.py`)
- Opens video with OpenCV, detects face with MediaPipe Face Mesh.
- Extracts forehead ROI green-channel signal over time.
- Band-pass filters and applies FFT to estimate dominant cardiac frequency.
- Returns BPM, confidence, and risk label.
- **Fallback**: Uses Haar cascade if MediaPipe is unavailable.

### Audio Module (`models/audio_module.py`)
- Loads audio with Librosa, computes energy envelope.
- Finds peaks in smoothed envelope to count breathing cycles.
- Derives breaths-per-minute estimate.
- Extracts ZCR, spectral centroid, and MFCC features for future classifier use.
- **Fallback**: Rule-based analysis always works even without pretrained models.

### Symptom Module (`models/symptom_module.py`)
- Runs a DistilBERT sentiment pipeline as a severity-tone proxy.
- Parallel keyword extraction against a curated symptom-severity database.
- Detects high-caution symptom pairs (e.g. chest pain + breathlessness).
- **Fallback**: Pure keyword rules if the transformer cannot be loaded.

### Score Engine (`models/score_engine.py`)
- Normalises each module output to a 0-100 sub-score.
- Weighted fusion (configurable, default 40/30/30) or trained ML fusion model.
- Cross-module clinical validation and symptom severity escalation.
- Confidence calibration based on scan quality and model source.
- Automatically rebalances if a module is missing.
- Generates contextual recommendations with clinical safety notes.

---

## Pretrained Model Setup

Vita AI supports optional pretrained deep-learning models that enhance accuracy.
Without them, all modules fall back to classical signal-processing pipelines.

### Auto-Downloadable Models

The following models are **automatically downloaded** on first use (or at server startup) and cached in `models_cache/`:

| Model | Source | Cache Directory | Enable Flag |
|-------|--------|-----------------|-------------|
| **DistilBERT** | HuggingFace (`distilbert-base-uncased-finetuned-sst-2-english`) | `models_cache/distilbert/` | `VITA_ENABLE_DISTILBERT` |
| **BioBERT** | HuggingFace (`dmis-lab/biobert-base-cased-v1.1`) | `models_cache/biobert/` | `VITA_ENABLE_BIOBERT` |
| **YAMNet** | TF-Hub (`google/yamnet/1`) | `models_cache/yamnet_saved_model/` | `VITA_ENABLE_YAMNET` |

Auto-download is enabled by default. Disable with `VITA_AUTO_DOWNLOAD_MODELS=false`.

#### CLI Download Scripts

Pre-download all auto-downloadable models:

```bash
python scripts/download_supported_models.py          # DistilBERT + BioBERT + YAMNet
python scripts/download_supported_models.py --model distilbert
python scripts/download_supported_models.py --model biobert
python scripts/download_supported_models.py --model yamnet
```

### Model Priority / Blending

**rPPG (Face Pipeline):**
The system uses Open-rPPG as the primary rPPG model. If it agrees with the classical
signal pipeline (within 10 BPM), an SNR-aware blend is used. If Open-rPPG is not
available, the classical signal-processing pipeline is used as fallback.

**Audio Pipeline:**
YAMNet is the primary audio classifier. Results are combined with the baseline
Librosa analysis. If YAMNet is not loaded, Librosa-only analysis provides the result.

**Symptom Pipeline:**
DistilBERT provides sentiment-based severity scoring. BioBERT provides clinical NER.
Both are optional; keyword-based rules always run as a fallback.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VITA_EFFICIENTPHYS_MODEL_PATH` | EfficientPhys weights (`.pth`) | — |
| `VITA_PHYSFORMER_MODEL_PATH` | PhysFormer weights (`.pth`) | — |
| `VITA_FACEPHYS_MODEL_PATH` | FacePhys weights (`.pth`) | — |
| `VITA_PHYSMAMBA_MODEL_PATH` | PhysMamba weights (`.pth`) | — |
| `VITA_TSCAN_MODEL_PATH` | (deprecated) TS-CAN weights | — |
| `VITA_PHYSNET_MODEL_PATH` | (deprecated) PhysNet weights | — |
| `VITA_DEEPPHYS_MODEL_PATH` | (deprecated) DeepPhys weights | — |
| `VITA_VGGISH_MODEL_PATH` | VGGish weights (`.pth`) | — |
| `VITA_YAMNET_MODEL_PATH` | YAMNet SavedModel dir or `"tfhub"` | — |
| `VITA_BIOBERT_MODEL` | BioBERT HF model ID or local path | — |
| `VITA_FUSION_MODEL_PATH` | Score fusion model (`.json` / `.pkl`) | — |
| `VITA_ENABLE_OPEN_RPPG` | Enable Open-rPPG as primary rPPG backend | `true` |
| `VITA_OPEN_RPPG_MODEL` | Open-rPPG model name (e.g. `PhysFormer.rlap`) | `FacePhys.rlap` |
| `VITA_AUTO_DOWNLOAD_MODELS` | Auto-download NLP/audio models | `true` |
| `VITA_AUTO_DOWNLOAD_MANUAL_MODELS` | Auto-download VGGish at startup | `true` |
| `VITA_AUTO_DOWNLOAD_RPPG_MODELS` | Check rPPG weights at startup | `true` |
| `VITA_PRELOAD_MODELS` | Preload all models at startup | `false` |
| `VITA_ENABLE_DISTILBERT` | Enable DistilBERT | `true` |
| `VITA_ENABLE_BIOBERT` | Enable BioBERT | `true` |
| `VITA_ENABLE_YAMNET` | Enable YAMNet | `true` |
| `VITA_ENABLE_VGGISH` | Enable VGGish | `true` |
| `VITA_ENABLE_EFFICIENTPHYS` | Enable EfficientPhys | `true` |
| `VITA_ENABLE_PHYSFORMER` | Enable PhysFormer | `true` |
| `VITA_ENABLE_FACEPHYS` | Enable FacePhys | `true` |
| `VITA_ENABLE_PHYSMAMBA` | Enable PhysMamba | `true` |
| `VITA_ENABLE_TSCAN` | Enable TS-CAN (deprecated) | `false` |
| `VITA_ENABLE_PHYSNET` | Enable PhysNet (deprecated) | `false` |
| `VITA_ENABLE_DEEPPHYS` | Enable DeepPhys (deprecated) | `false` |

### Checking Model Status

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/models
```

---

## Training a Fusion Model

```bash
python -m train.train_fusion_model \
    --data train_data.csv \
    --output models/fusion_trained.json \
    --backend xgboost
```

CSV columns: `heart_score, breathing_score, symptom_score, conf_heart, conf_breathing, conf_symptom, label`

Then set `VITA_FUSION_MODEL_PATH=models/fusion_trained.json` and restart the server.

---

## Evaluation

```bash
# rPPG evaluation (requires benchmark dataset)
python eval/eval_rppg.py --dataset ubfc --path datasets/ubfc

# Audio evaluation
python eval/eval_audio.py --data eval/datasets/audio_labels.csv

# Symptom evaluation
python eval/eval_symptom.py --data eval/datasets/symptom_labels.csv

# Run all at once
python eval/run_all_evals.py \
    --rppg-dataset ubfc --rppg-path datasets/ubfc \
    --audio-data eval/datasets/audio_labels.csv \
    --symptom-data eval/datasets/symptom_labels.csv
```

Results saved to `eval/results/`.

---

## Future Integration Points

| Area | Where to extend |
|------|----------------|
| **Flutter / mobile app** | Call REST endpoints; use `multipart/form-data` for files |
| **Database** | Add a DB session dependency in `api/main.py`; persist module outputs |
| **Authentication** | Add OAuth2 / JWT middleware in the FastAPI app |
| **Real-time streaming** | Add a WebSocket endpoint for camera/audio frame streaming |
| **Dashboard analytics** | Read `component_scores` and `vita_health_score` trends from DB |
| **Model swapping** | Replace module internals at marked `EXTENSION POINT` comments |
| **Cloud deployment** | Containerise with Docker; deploy on AWS / GCP / Azure |

---

## Limitations

- **rPPG accuracy** depends on lighting, camera quality, and subject stillness. Open-rPPG provides pretrained deep models (FacePhys, EfficientPhys, PhysFormer, etc.) that significantly improve accuracy over classical signal processing.
- **Breathing-rate estimation** from audio energy peaks is approximate. Enable YAMNet/VGGish for respiratory sound classification.
- **Symptom analysis** uses sentiment as a severity proxy by default. BioBERT provides more clinically relevant classification when loaded.
- The Vita Health Score uses a weighted sum by default — train a fusion model with `train/train_fusion_model.py` for better accuracy.
- Authentication uses simple SHA-256 password hashing — use bcrypt/Argon2 for production.
- No rate limiting is included in this prototype.

---

## License

This project is provided as-is for hackathon / educational purposes.
