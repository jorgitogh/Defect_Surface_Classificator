from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    raw_neu_dir: Path
    raw_images_dir: Path
    raw_annotations_dir: Path
    processed_metadata_dir: Path
    processed_splits_dir: Path
    runs_dir: Path


def get_paths() -> Paths:
    # ml/src/config.py -> project root is two levels up
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_neu_dir = raw_dir / "neu"
    raw_images_dir = raw_neu_dir / "images"
    raw_annotations_dir = raw_neu_dir / "annotations"

    processed_metadata_dir = processed_dir / "metadata"
    processed_splits_dir = processed_dir / "splits"

    runs_dir = project_root / "ml" / "runs"

    # Ensure output dirs exist
    processed_metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_splits_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_neu_dir=raw_neu_dir,
        raw_images_dir=raw_images_dir,
        raw_annotations_dir=raw_annotations_dir,
        processed_metadata_dir=processed_metadata_dir,
        processed_splits_dir=processed_splits_dir,
        runs_dir=runs_dir,
    )
