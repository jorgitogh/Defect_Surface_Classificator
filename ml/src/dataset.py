from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ml.src.config import get_paths


@dataclass(frozen=True)
class SplitData:
    df: pd.DataFrame
    root_raw_neu: Path
    num_classes: int


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class NeuDefectDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        paths = get_paths()
        split_csv = paths.processed_metadata_dir / "split.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"split.csv not found at {split_csv}. Run prepare_data.py first.")

        df = pd.read_csv(split_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for split='{split}' in {split_csv}")

        self.df = df
        self.root = paths.raw_neu_dir
        self.transform = get_transforms(split)
        self.num_classes = int(df["label_id"].nunique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / row["rel_path"]
        label_id = int(row["label_id"])

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y
