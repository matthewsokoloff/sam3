#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_split(path: Path) -> set[str]:
    ids = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line:
            ids.add(Path(line).stem)

    return ids


def get_objects(annotation: dict) -> list:
    for key in ("objects", "annotations"):
        value = annotation.get(key)

        if isinstance(value, list):
            return value

    return []


def get_label(obj: dict) -> str | None:
    for key in (
        "label",
        "key",
        "class",
        "category",
        "category_name",
    ):
        value = obj.get(key)

        if isinstance(value, str) and value:
            return value

    return None


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}

    return bool(value)


def get_flag(obj: dict, *names: str) -> bool:
    properties = obj.get("properties", {})

    if not isinstance(properties, dict):
        properties = {}

    for name in names:
        if name in properties:
            return as_bool(properties[name])

        if name in obj:
            return as_bool(obj[name])

    return False


def get_bbox(obj: dict):
    bbox = obj.get("bbox")

    if isinstance(bbox, dict):
        xmin = bbox.get("xmin", bbox.get("x_min"))
        ymin = bbox.get("ymin", bbox.get("y_min"))
        xmax = bbox.get("xmax", bbox.get("x_max"))
        ymax = bbox.get("ymax", bbox.get("y_max"))

        if None not in (xmin, ymin, xmax, ymax):
            return tuple(float(v) for v in (xmin, ymin, xmax, ymax))

        x = bbox.get("x")
        y = bbox.get("y")
        width = bbox.get("width", bbox.get("w"))
        height = bbox.get("height", bbox.get("h"))

        if None not in (x, y, width, height):
            return (
                float(x),
                float(y),
                float(x) + float(width),
                float(y) + float(height),
            )

    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(float(v) for v in bbox)

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "data/mtsd/annotations/"
            "mtsd_v2_fully_annotated"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mtsd_top15.json"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtsd/subset/metadata"),
    )

    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    config = json.loads(
        args.config.read_text(encoding="utf-8")
    )

    class_entries = config["classes"]
    generic_prompt = config["generic_prompt"]

    labels = [entry["label"] for entry in class_entries]
    prompts = {
        entry["label"]: entry["prompt"]
        for entry in class_entries
    }

    annotation_dir = args.root / "annotations"
    split_dir = args.root / "splits"

    train_ids = read_split(split_dir / "train.txt")
    val_ids = read_split(split_dir / "val.txt")
    allowed_ids = train_ids | val_ids

    print("Train IDs:", len(train_ids))
    print("Validation IDs:", len(val_ids))
    print("Allowed train+val IDs:", len(allowed_ids))

    candidates = defaultdict(list)
    unreadable = 0
    missing_bbox = 0
    skipped_dummy = 0
    skipped_ambiguous = 0

    annotation_files = sorted(annotation_dir.glob("*.json"))

    print("Annotation files:", len(annotation_files))

    for path in annotation_files:
        image_id = path.stem

        if image_id not in allowed_ids:
            continue

        try:
            annotation = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            unreadable += 1
            continue

        width = annotation.get(
            "width",
            annotation.get("image_width"),
        )
        height = annotation.get(
            "height",
            annotation.get("image_height"),
        )

        split = "train" if image_id in train_ids else "val"

        for object_index, obj in enumerate(
            get_objects(annotation)
        ):
            if not isinstance(obj, dict):
                continue

            label = get_label(obj)

            if label not in prompts:
                continue

            dummy = get_flag(obj, "dummy")
            ambiguous = get_flag(obj, "ambiguous")

            if dummy:
                skipped_dummy += 1
                continue

            if ambiguous:
                skipped_ambiguous += 1
                continue

            bbox = get_bbox(obj)

            if bbox is None:
                missing_bbox += 1
                continue

            xmin, ymin, xmax, ymax = bbox

            box_width = max(0.0, xmax - xmin)
            box_height = max(0.0, ymax - ymin)
            box_area = box_width * box_height

            relative_area = ""

            if width and height:
                image_area = float(width) * float(height)

                if image_area > 0:
                    relative_area = box_area / image_area

            candidates[label].append(
                {
                    "image_id": image_id,
                    "split": split,
                    "annotation_path": str(path.resolve()),
                    "class_label": label,
                    "fine_prompt": prompts[label],
                    "generic_prompt": generic_prompt,
                    "object_index": object_index,
                    "image_width": width or "",
                    "image_height": height or "",
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_area": box_area,
                    "relative_area": relative_area,
                    "occluded": get_flag(obj, "occluded"),
                    "out_of_frame": get_flag(
                        obj,
                        "out-of-frame",
                        "out_of_frame",
                    ),
                }
            )

    print("\nUsable candidates:")

    for label in labels:
        print(f"{len(candidates[label]):>6}  {label}")

    # Process rarer classes first so common classes do not consume
    # images needed by rarer classes.
    selection_order = sorted(
        labels,
        key=lambda label: len(candidates[label]),
    )

    selected_by_label = defaultdict(list)
    used_images = set()

    for class_offset, label in enumerate(selection_order):
        rows = list(candidates[label])

        class_rng = random.Random(
            f"{args.seed}:{class_offset}:{label}"
        )
        class_rng.shuffle(rows)

        for row in rows:
            if row["image_id"] in used_images:
                continue

            selected_by_label[label].append(row)
            used_images.add(row["image_id"])

            if len(selected_by_label[label]) >= args.per_class:
                break

    selected_rows = []

    for class_rank, label in enumerate(labels, start=1):
        rows = selected_by_label[label]

        for sample_number, row in enumerate(rows, start=1):
            row = dict(row)
            row["class_rank"] = class_rank
            row["sample_number"] = sample_number
            row["sample_id"] = (
                f"class{class_rank:02d}_"
                f"sample{sample_number:03d}_"
                f"{row['image_id']}"
            )
            selected_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    selected_ids_path = (
        args.output_dir / "selected_image_ids.txt"
    )
    summary_path = args.output_dir / "class_summary.csv"

    fieldnames = [
        "sample_id",
        "class_rank",
        "sample_number",
        "image_id",
        "split",
        "class_label",
        "fine_prompt",
        "generic_prompt",
        "object_index",
        "annotation_path",
        "image_width",
        "image_height",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "box_width",
        "box_height",
        "box_area",
        "relative_area",
        "occluded",
        "out_of_frame",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    with selected_ids_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for image_id in sorted(used_images):
            handle.write(image_id + "\n")

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "class_rank",
                "class_label",
                "fine_prompt",
                "candidate_instances",
                "selected_instances",
            ]
        )

        for class_rank, label in enumerate(labels, start=1):
            writer.writerow(
                [
                    class_rank,
                    label,
                    prompts[label],
                    len(candidates[label]),
                    len(selected_by_label[label]),
                ]
            )

    print("\nSelected per class:")

    for label in labels:
        print(
            f"{len(selected_by_label[label]):>6}  {label}"
        )

    print("\nTotal selected rows:", len(selected_rows))
    print("Unique selected images:", len(used_images))
    print("Unreadable annotations:", unreadable)
    print("Missing bounding boxes:", missing_bbox)
    print("Skipped dummy objects:", skipped_dummy)
    print("Skipped ambiguous objects:", skipped_ambiguous)

    print("\nManifest:", manifest_path.resolve())
    print("Selected IDs:", selected_ids_path.resolve())
    print("Summary:", summary_path.resolve())


if __name__ == "__main__":
    main()
