from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from ml.src.config import get_paths
from ml.src.dataset import NeuDefectDataset
from ml.src.utils.seed import seed_everything
from ml.src.utils.metrics import compute_metrics


def build_model(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = sorted(list(set(y_true.tolist())))
    return compute_metrics(y_true, y_pred, labels=labels)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    paths = get_paths()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    train_ds = NeuDefectDataset(split="train")
    val_ds = NeuDefectDataset(split="val")
    num_classes = 6

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.runs_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | val_f1_macro={row['val_f1_macro']:.4f}"
        )

        if row["val_f1_macro"] > best_f1:
            best_f1 = row["val_f1_macro"]
            torch.save(model.state_dict(), best_path)
            print(f"[OK] New best model saved: {best_path} (val_f1_macro={best_f1:.4f})")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_f1_macro": best_f1}, f, indent=2)
    print(f"[OK] Wrote {metrics_path}")


if __name__ == "__main__":
    main()
