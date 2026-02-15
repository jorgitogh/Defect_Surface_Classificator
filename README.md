# Defect Surface Classifier

## Web Demo (Frontend + Backend)

This repository now includes:

- `api/`: FastAPI backend for image inference
- `web/`: static frontend page to upload an image and visualize probabilities

### 1. Backend setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r api/requirements.txt
```

Run API:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

The API auto-loads:

- latest `ml/runs/run_*/best_model.pt`
- `data/processed/metadata/class_map.json`

Optional environment variables:

- `MODEL_CHECKPOINT` to force a specific checkpoint path
- `MODEL_CHECKPOINT_URL` to download checkpoint from a URL at startup
- `CLASS_MAP_PATH` to force class map path
- `CORS_ORIGINS` comma-separated (default: `http://localhost:5173,http://127.0.0.1:5173`)

### 2. Frontend setup

In a second terminal from project root:

```powershell
python -m http.server 5173 --directory web
```

Open:

- `http://localhost:5173`

The frontend calls:

- `GET /classes`
- `POST /predict`

### 3. API endpoints

- `GET /health`
- `GET /classes`
- `POST /predict` with `multipart/form-data` field `file`

Example curl:

```powershell
curl -X POST http://127.0.0.1:8000/predict -F "file=@data/raw/neu/images/crazing/crazing_1.jpg"
```

## Deploy Free on Render (Blueprint)

This repo includes `render.yaml` for one-click deployment:

- `defect-classifier-api` (FastAPI web service)
- `defect-classifier-web` (static site)

### Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and select this repository.
3. During setup, provide `MODEL_CHECKPOINT_URL` for the API service.
4. Deploy both services.

The static frontend is configured at build time to call the deployed API URL automatically.

### Notes

- Do not commit model weights or secrets into git.
- Keep real values in Render environment variables.
- Free tier services can sleep after inactivity and cold-start on next request.
