from __future__ import annotations

import hashlib
import html
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

          :root {
            --bg-a: #f2f6fa;
            --bg-b: #dfe8f0;
            --panel: #ffffff;
            --text-main: #0f2133;
            --text-muted: #5e7489;
            --line: #c8d7e5;
            --accent: #e98b2a;
            --accent-2: #2f8ba9;
          }

          .stApp {
            background:
              radial-gradient(65rem 38rem at 102% -5%, #c7d9ea 0%, transparent 62%),
              radial-gradient(52rem 34rem at -6% -12%, #edf4fb 0%, transparent 50%),
              linear-gradient(135deg, var(--bg-a), var(--bg-b));
            color: var(--text-main);
          }

          .main .block-container {
            max-width: 1080px;
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
          }

          h1, h2, h3 {
            font-family: "Chakra Petch", sans-serif !important;
            letter-spacing: 0.01em;
            color: var(--text-main);
          }

          .hero-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.15rem 1.1rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(245,250,255,0.95));
            box-shadow: 0 12px 28px rgba(22, 47, 75, 0.08);
            margin-bottom: 1rem;
          }

          .hero-card .eyebrow {
            margin: 0;
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.72rem;
            color: var(--accent-2);
          }

          .hero-card h1 {
            margin: 0.25rem 0 0.3rem;
            font-size: clamp(1.6rem, 3vw, 2.4rem);
            line-height: 1.04;
          }

          .hero-card .subtitle {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.95rem;
          }

          .status-badge {
            display: inline-block;
            margin: 0.65rem 0 0.3rem;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            background: #edf6fb;
            border: 1px solid #c6deec;
            color: #2f6f8a;
            font-size: 0.75rem;
            font-family: "IBM Plex Mono", monospace;
          }

          .image-card,
          .results-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--panel);
            box-shadow: 0 10px 24px rgba(22, 47, 75, 0.07);
            padding: 0.85rem;
          }

          .image-caption {
            margin-top: 0.45rem;
            margin-bottom: 0;
            color: var(--text-muted);
            font-size: 0.84rem;
          }

          .top-prediction {
            border: 1px solid #dce7f1;
            border-radius: 13px;
            padding: 0.8rem;
            background: linear-gradient(150deg, #f8fbff, #edf5fc);
            margin-bottom: 0.75rem;
          }

          .top-prediction .label {
            margin: 0 0 0.28rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.68rem;
            font-family: "IBM Plex Mono", monospace;
            color: #63829a;
          }

          .top-prediction .class-name {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 700;
            color: #16344f;
            line-height: 1.15;
          }

          .top-prediction .confidence {
            margin: 0.2rem 0 0;
            font-size: 0.9rem;
            color: #2f8ba9;
            font-family: "IBM Plex Mono", monospace;
          }

          .prob-row {
            margin-bottom: 0.6rem;
          }

          .prob-label-line {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.23rem;
          }

          .prob-name {
            color: #29445d;
            font-size: 0.9rem;
          }

          .prob-value {
            color: #5f7990;
            font-size: 0.8rem;
            font-family: "IBM Plex Mono", monospace;
          }

          .prob-track {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: #e5edf4;
            overflow: hidden;
          }

          .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            transition: width 320ms ease;
          }

          .section-note {
            color: var(--text-muted);
            font-size: 0.83rem;
            margin-top: 0.35rem;
          }

          @media (max-width: 900px) {
            .main .block-container {
              padding-top: 1.1rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret_or_env(name: str) -> str:
    from_env = os.getenv(name, "").strip()
    if from_env:
        return from_env
    try:
        from_secret = str(st.secrets.get(name, "")).strip()
    except Exception:
        from_secret = ""
    return from_secret


def resolve_class_map_path(paths) -> Path:
    class_map_value = get_secret_or_env("CLASS_MAP_PATH")
    if class_map_value:
        class_map_path = Path(class_map_value).expanduser().resolve()
    else:
        class_map_path = paths.processed_metadata_dir / "class_map.json"
    return class_map_path


def resolve_checkpoint_path(paths) -> Path:
    ckpt_url = get_secret_or_env("MODEL_CHECKPOINT_URL")
    if ckpt_url:
        return download_checkpoint(ckpt_url.strip())

    ckpt_local = get_secret_or_env("MODEL_CHECKPOINT")
    if ckpt_local:
        checkpoint_path = Path(ckpt_local).expanduser().resolve()
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Configured MODEL_CHECKPOINT does not exist: {checkpoint_path}")

    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt is not None:
        return latest_ckpt

    raise FileNotFoundError(
        "No checkpoint found. Configure MODEL_CHECKPOINT_URL (recommended) "
        "or MODEL_CHECKPOINT in environment/secrets."
    )


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


def pretty_class_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def render_top_prediction(top_class: str, top_prob: float) -> None:
    safe_class = html.escape(pretty_class_name(top_class))
    st.markdown(
        f"""
        <div class="top-prediction">
          <p class="label">Top Prediction</p>
          <p class="class-name">{safe_class}</p>
          <p class="confidence">{top_prob * 100:.2f}% confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_rows(rows: list[tuple[str, float]]) -> None:
    for class_name, prob in rows:
        safe_class = html.escape(pretty_class_name(class_name))
        width = max(0.0, min(prob, 1.0)) * 100.0
        st.markdown(
            f"""
            <div class="prob-row">
              <div class="prob-label-line">
                <span class="prob-name">{safe_class}</span>
                <span class="prob-value">{prob * 100:.2f}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" style="width:{width:.2f}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Defect Surface Classifier", layout="wide")
    inject_styles()

    st.markdown(
        """
        <section class="hero-card">
          <p class="eyebrow">Computer Vision Showcase</p>
          <h1>Surface Defect Classifier</h1>
          <p class="subtitle">Upload a steel surface image to predict the defect type and inspect model confidence.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    paths = get_paths()

    try:
        class_names = load_class_names(resolve_class_map_path(paths))
    except Exception as exc:
        st.error(f"Configuration error (class map): {exc}")
        st.stop()

    try:
        checkpoint_path = resolve_checkpoint_path(paths)
    except Exception as exc:
        st.error(f"Configuration error (checkpoint): {exc}")
        st.stop()

    try:
        model, device = load_model(str(checkpoint_path), len(class_names))
    except Exception as exc:
        st.error(f"Model loading error: {exc}")
        st.stop()

    st.markdown(f'<span class="status-badge">Model ready on {device}</span>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 1.0], gap="large")

    with left_col:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp", "webp"])

        image = None
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
            except UnidentifiedImageError:
                st.error("Could not open image file.")

        if image is not None:
            st.image(image, use_container_width=True)
            st.markdown(
                f'<p class="image-caption">{html.escape(uploaded.name)}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload one image to run prediction.")

        run_prediction = st.button("Analyze Surface", type="primary", use_container_width=True)
        if run_prediction:
            if image is None:
                st.warning("Please upload a valid image first.")
            else:
                rows = predict_image(model, device, image, class_names)
                st.session_state["last_rows"] = rows
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        if "last_rows" in st.session_state:
            rows = st.session_state["last_rows"]
            top_class, top_prob = rows[0]
            render_top_prediction(top_class, top_prob)
            st.markdown("#### Probability Breakdown")
            render_probability_rows(rows)
            st.markdown(
                '<p class="section-note">Higher bars indicate stronger model confidence for that class.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("#### Results")
            st.info("Your prediction output will appear here after analyzing an image.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
