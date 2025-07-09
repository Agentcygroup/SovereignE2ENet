# SovereignE2ENet

**Fully Sovereign AI Stack (Mac M1 Optimized)**

## Key Features
- No cloud calls. No telemetry.
- Local model training + inference using MPS (Apple Silicon GPU)
- REST API built with FastAPI
- Full training/eval pipeline

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers fastapi uvicorn
```

Run training:
```bash
python scripts/train.py
```

Run eval:
```bash
python eval/eval.py
```

Run API server:
```bash
uvicorn api.main:app --reload
```

