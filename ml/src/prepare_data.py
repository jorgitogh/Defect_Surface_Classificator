from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.config import get_paths


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def parse_voc_xml(xml_path: Path) -> dict:
    """
    Parse Pascal VOC style XML.
    Returns dict with filename, label, width, height, depth, bbox.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", default="")

    size = root.find("size")
    width = int(size.findtext("width", default="0")) if size is not None else 0
    height = int(size.findtext("height", default="0")) if size is not None else 0
    depth = int(size.findtext("depth", default="0")) if size is not None else 0

    obj = root.find("object")
    label = obj.findtext("name", default="") if obj is not None else ""

    bbox = obj.find("bndbox") if obj is not None else None
    xmin = int(bbox.findtext("xmin", default="0")) if bbox is not None else 0
    ymin = int(bbox.findtext("ymin", default="0")) if bbox is not None else 0
    xmax = int(bbox.findtext("xmax", default="0")) if bbox is not None else 0
    ymax = int(bbox.findtext("ymax", default="0")) if bbox is not None else 0

    return {
        "filename": filename,
        "label": label,
        "width": width,
        "height": height,
        "depth": depth,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def find_image_path(images_root: Path, label: str, filename: str) -> Path | None:
    candidate = images_root / label / filename
    if candidate.exists():
        return candidate

    matches = list(images_root.rglob(filename))
    if matches:
        return matches[0]
    return None


def build_metadata(images_root: Path, annotations_root: Path) -> pd.DataFrame:
    rows = []

    xml_files = sorted(annotations_root.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {annotations_root}")

    for xml_path in xml_files:
        info = parse_voc_xml(xml_path)
        label = info["label"]
        filename = info["filename"]

        img_path = find_image_path(images_root, label, filename)
        if img_path is None:
            rel_img = None
            exists = 0
        else:
            rel_img = img_path.relative_to(images_root.parent) 
            exists = 1

        w, h = info["width"], info["height"]
        xmin, ymin, xmax, ymax = info["xmin"], info["ymin"], info["xmax"], info["ymax"]
        bbox_w = max(0, xmax - xmin)
        bbox_h = max(0, ymax - ymin)
        bbox_area = bbox_w * bbox_h
        img_area = max(1, w * h) if w and h else 1
        bbox_area_ratio = bbox_area / img_area

        rows.append(
            {
                "rel_path": str(rel_img) if rel_img else "",
                "filename": filename,
                "label": label,
                "width": w,
                "height": h,
                "depth": info["depth"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "bbox_area": bbox_area,
                "bbox_area_ratio": bbox_area_ratio,
                "xml_rel_path": str(xml_path.relative_to(images_root.parent)),
                "image_exists": exists,
            }
        )

    df = pd.DataFrame(rows)

    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)

    missing = int((df["image_exists"] == 0).sum())
    if missing:
        print(f"[WARN] Missing images for {missing} annotations. They will be dropped.")
    df = df[df["image_exists"] == 1].reset_index(drop=True)

    return df


def make_splits(df: pd.DataFrame, seed: int = 42, train_size: float = 0.7, val_size: float = 0.15):
    assert 0 < train_size < 1
    assert 0 < val_size < 1
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=df["label"],
    )

    val_prop = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_prop),
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return split_df


def main():
    paths = get_paths()

    if not paths.raw_images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {paths.raw_images_dir}")
    if not paths.raw_annotations_dir.exists():
        raise FileNotFoundError(
            f"Annotations folder not found: {paths.raw_annotations_dir}\n"
            "Rename your folder to data/raw/neu/annotations (currently you have 'annotation'?)"
        )


    class_map = {name: i for i, name in enumerate(CLASSES)}

    df = build_metadata(paths.raw_images_dir, paths.raw_annotations_dir)
    df["label_id"] = df["label"].map(class_map).astype(int)

    images_csv = paths.processed_metadata_dir / "images.csv"
    df.to_csv(images_csv, index=False)
    print(f"[OK] Wrote {images_csv} with {len(df)} rows.")


    split_df = make_splits(df, seed=42, train_size=0.7, val_size=0.15)
    split_csv = paths.processed_metadata_dir / "split.csv"
    split_df[["rel_path", "label", "label_id", "split"]].to_csv(split_csv, index=False)
    print(f"[OK] Wrote {split_csv}.")

    class_map_path = paths.processed_metadata_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {class_map_path}.")


    for split_name in ["train", "val", "test"]:
        out_txt = paths.processed_splits_dir / f"{split_name}.txt"
        lines = split_df.loc[split_df["split"] == split_name, "rel_path"].tolist()
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Wrote {out_txt} ({len(lines)} lines).")

    print("\n=== Class distribution (all) ===")
    print(df["label"].value_counts().to_string())
    print("\n=== Class distribution by split ===")
    print(split_df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
