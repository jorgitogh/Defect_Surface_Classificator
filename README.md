# Defect Surface Classifier

## Streamlit Demo (Simple Option)

This is the easiest way to present your model online or locally.

### 1. Install dependencies

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run locally

```powershell
streamlit run app.py
```

The app lets you choose:

- local checkpoint path (`best_model.pt`)
- or checkpoint URL (for example Hugging Face direct `.../resolve/main/best_model.pt`)

It reads classes from:

- `data/processed/metadata/class_map.json`

### 3. Deploy for free on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create new app using:
   - Repository: your repo
   - Branch: your branch
   - Main file path: `app.py`
4. In app settings, add secrets (optional but recommended):

```toml
MODEL_CHECKPOINT_URL = "https://huggingface.co/<user>/<repo>/resolve/main/best_model.pt"
```

Then in the app sidebar choose `Checkpoint source = URL` and use that value.

### Notes

- Do not commit model weights (`.pt`) or private keys to git.
- If URL loading fails, verify it is a direct file URL and not an HTML page.
