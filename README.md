# Defect Surface Classifier

## What This Code Does

This project classifies steel surface defects using a ResNet18 model.

Main components:

- `ml/src/prepare_data.py`: parses annotations, builds metadata, and creates train/val/test splits.
- `ml/src/train.py`: trains the model and saves the best checkpoint (`best_model.pt`).
- `ml/src/evaluate.py`: evaluates a trained checkpoint and writes metrics/reports.
- `app.py`: Streamlit app for interactive image inference (upload image, get prediction + probabilities).

Class mapping is stored in:

- `data/processed/metadata/class_map.json`

## How To Use It

### 1. Install dependencies

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure model source (recommended)

The app has no user-facing model setup. It loads config automatically in this order:

1. `MODEL_CHECKPOINT_URL`
2. `MODEL_CHECKPOINT`
3. Latest local checkpoint in `ml/runs/run_*/best_model.pt`

Optional local configuration methods:

1. Environment variables in PowerShell:

```powershell
$env:MODEL_CHECKPOINT_URL="https://huggingface.co/<user>/<repo>/resolve/main/best_model.pt"
```

2. Local secrets file at `.streamlit/secrets.toml` (or `%USERPROFILE%\.streamlit\secrets.toml`):

```toml
MODEL_CHECKPOINT_URL = "https://huggingface.co/<user>/<repo>/resolve/main/best_model.pt"
```

Optional class map override:

```toml
CLASS_MAP_PATH = "data/processed/metadata/class_map.json"
```

### 3. Run the Streamlit app locally

```powershell
python -m streamlit run app.py
```


