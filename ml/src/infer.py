from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from ml.src.config import get_paths


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    class_name: str


class DefectPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        class_map_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.class_map_path = class_map_path or (get_paths().processed_metadata_dir / "class_map.json")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.classes = self._load_classes(self.class_map_path)
        self.model = self._build_model(num_classes=len(self.classes))
        state_dict = self._load_checkpoint_state_dict(self.checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_classes(class_map_path: Path) -> list[ClassInfo]:
        if not class_map_path.exists():
            raise FileNotFoundError(f"class_map.json not found at {class_map_path}")

        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)

        items = sorted(class_map.items(), key=lambda item: int(item[1]))
        return [ClassInfo(class_id=int(class_id), class_name=str(class_name)) for class_name, class_id in items]

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def _load_checkpoint_state_dict(self, checkpoint_path: Path) -> dict:
        try:
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=self.device)
        except UnpicklingError as exc:
            raise RuntimeError(
                "Could not read checkpoint. The file is likely not a valid PyTorch checkpoint "
                "(commonly an HTML page from a non-direct MODEL_CHECKPOINT_URL)."
            ) from exc

        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if isinstance(payload, dict):
            return payload

        raise RuntimeError(
            "Checkpoint did not contain a state_dict dictionary. "
            "Export with torch.save(model.state_dict(), 'best_model.pt')."
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        if image.mode != "RGB":
            image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_index = int(np.argmax(probabilities))
        top_class = self.classes[top_index]
        top_confidence = float(probabilities[top_index])

        per_class = [
            {
                "class_id": info.class_id,
                "class_name": info.class_name,
                "probability": float(probabilities[idx]),
            }
            for idx, info in enumerate(self.classes)
        ]
        per_class.sort(key=lambda row: row["probability"], reverse=True)

        return {
            "top_class_id": top_class.class_id,
            "top_class_name": top_class.class_name,
            "top_confidence": top_confidence,
            "probabilities": per_class,
        }
