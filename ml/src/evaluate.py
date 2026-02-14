from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from ml.src.config import get_paths
from ml.src.dataset import NeuDefectDataset
from ml.src.utils.metrics import compute_metrics, make_classification_report


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def build_model(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(preds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    paths = get_paths()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    test_ds = NeuDefectDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=6).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_pred = predict(model, test_loader, device)
    labels = list(range(6))
    metrics = compute_metrics(y_true, y_pred, labels=labels)

    print("\n=== TEST METRICS ===")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro F1:   {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:{metrics['f1_weighted']:.4f}")

    report = make_classification_report(y_true, y_pred, target_names=CLASSES)
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    out_dir = ckpt_path.parent
    (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {out_dir / 'test_report.txt'}")


    out_metrics = out_dir / "test_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"[OK] Wrote {out_metrics}")


if __name__ == "__main__":
    main()
