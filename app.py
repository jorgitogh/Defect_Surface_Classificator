from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from pickle import UnpicklingError
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

from ml.src.config import get_paths


def get_secret_or_env(name: str) -> str:
    from_env = os.getenv(name, "").strip()
    if from_env:
        return from_env
    try:
        from_secret = str(st.secrets.get(name, "")).strip()
    except Exception:
        from_secret = ""
    return from_secret


def find_latest_checkpoint() -> Path | None:
    paths = get_paths()
    run_dirs = sorted(
        [p for p in paths.runs_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidate = run_dir / "best_model.pt"
        if candidate.exists():
            return candidate
    return None


def download_checkpoint(url: str) -> Path:
    if not url:
        raise ValueError("Checkpoint URL is empty.")

    cache_dir = Path(tempfile.gettempdir()) / "defect_surface_streamlit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".pt"
    checkpoint_path = cache_dir / filename
    if checkpoint_path.exists():
        return checkpoint_path

    request = Request(url, headers={"User-Agent": "defect-surface-streamlit/1.0"})
    try:
        with urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"Download failed with HTTP status {response.status}.")
            payload = response.read()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "text/html" in content_type:
                raise RuntimeError(
                    "The URL returned HTML, not checkpoint bytes. "
                    "Use a direct file URL (Hugging Face: .../resolve/main/best_model.pt)."
                )
            checkpoint_path.write_bytes(payload)
    except URLError as exc:
        raise RuntimeError(f"Could not download checkpoint from URL: {exc}") from exc

    return checkpoint_path


def load_class_names(class_map_path: Path) -> list[str]:
    if not class_map_path.exists():
        raise FileNotFoundError(f"class_map.json not found at {class_map_path}")
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    ordered = sorted(class_map.items(), key=lambda item: int(item[1]))
    return [name for name, _ in ordered]


def load_state_dict_compatible(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    except UnpicklingError as exc:
        raise RuntimeError(
            "Checkpoint file is not a valid PyTorch state_dict (possible HTML download or corrupted file)."
        ) from exc

    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError("Unsupported checkpoint format. Expected a state_dict dictionary.")


@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = load_state_dict_compatible(Path(checkpoint_path), device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predict_image(model, device, image: Image.Image, class_names: list[str]) -> list[tuple[str, float]]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = get_transform()(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    rows = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def main():
    st.set_page_config(page_title="Defect Surface Classifier", page_icon=":mag:", layout="centered")
    st.title("Defect Surface Classifier")
    st.caption("Simple Streamlit demo for image classification with your trained ResNet18 checkpoint.")

    paths = get_paths()
    class_map_default = paths.processed_metadata_dir / "class_map.json"
    latest_ckpt = find_latest_checkpoint()

    with st.sidebar:
        st.subheader("Model Setup")
        source = st.radio("Checkpoint source", options=["Local path", "URL"], index=0)
        ckpt_local_default = get_secret_or_env("MODEL_CHECKPOINT") or (str(latest_ckpt) if latest_ckpt else "")
        ckpt_url_default = get_secret_or_env("MODEL_CHECKPOINT_URL")
        class_map_input = st.text_input("class_map.json path", value=str(class_map_default))
        if source == "Local path":
            ckpt_input = st.text_input("Checkpoint path (.pt)", value=ckpt_local_default)
        else:
            ckpt_input = st.text_input("Checkpoint URL", value=ckpt_url_default)

    try:
        class_names = load_class_names(Path(class_map_input))
    except Exception as exc:
        st.error(f"Class map error: {exc}")
        st.stop()

    try:
        if source == "URL":
            checkpoint_path = download_checkpoint(ckpt_input.strip())
        else:
            checkpoint_path = Path(ckpt_input).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    except Exception as exc:
        st.error(f"Checkpoint setup error: {exc}")
        st.stop()

    try:
        model, device = load_model(str(checkpoint_path), len(class_names))
    except Exception as exc:
        st.error(f"Model loading error: {exc}")
        st.stop()

    st.success(f"Model loaded on `{device}`")
    st.caption(f"Checkpoint: `{checkpoint_path}`")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("Upload one image to run prediction.")
        return

    try:
        image = Image.open(uploaded)
    except UnidentifiedImageError:
        st.error("Could not open image file.")
        return

    st.image(image, caption=uploaded.name, use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        rows = predict_image(model, device, image, class_names)
        top_class, top_prob = rows[0]
        st.metric(label="Top prediction", value=top_class, delta=f"{top_prob * 100:.2f}% confidence")

        st.subheader("Class probabilities")
        table_rows = [{"class": name, "probability": prob} for name, prob in rows]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        chart_data = {name: prob for name, prob in rows}
        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
