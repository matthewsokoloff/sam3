#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def get_objects(data):
    if isinstance(data, dict):
        if isinstance(data.get("objects"), list):
            return data["objects"]
        if isinstance(data.get("annotations"), list):
            return data["annotations"]
    return []


def get_label(obj):
    for key in ("label", "class", "category", "category_name"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
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
        "--output",
        type=Path,
        default=Path("data/mtsd/results/class_counts.csv"),
    )
    args = parser.parse_args()

    annotation_dir = args.root / "annotations"

    if not annotation_dir.exists():
        raise SystemExit(
            f"Annotation folder not found: {annotation_dir}"
        )

    files = sorted(annotation_dir.glob("*.json"))

    print("Annotation directory:", annotation_dir)
    print("Annotation JSON files:", len(files))

    label_counts = Counter()
    images_with_objects = 0
    total_objects = 0
    unreadable_files = 0
    sample_data = None
    sample_path = None

    for path in files:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            unreadable_files += 1
            print(f"Could not read {path.name}: {exc}")
            continue

        objects = get_objects(data)

        if objects:
            images_with_objects += 1

        if sample_data is None and objects:
            sample_data = data
            sample_path = path

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            label = get_label(obj)

            if label is None:
                label = "<missing-label>"

            label_counts[label] += 1
            total_objects += 1

    print("Images with annotated objects:", images_with_objects)
    print("Total annotated objects:", total_objects)
    print("Unique labels:", len(label_counts))
    print("Unreadable files:", unreadable_files)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "label", "instance_count"])

        for rank, (label, count) in enumerate(
            label_counts.most_common(),
            start=1,
        ):
            writer.writerow([rank, label, count])

    print("\nTop 40 labels:")

    for rank, (label, count) in enumerate(
        label_counts.most_common(40),
        start=1,
    ):
        print(f"{rank:>3}. {count:>7}  {label}")

    print("\nClass counts saved to:")
    print(args.output.resolve())

    if sample_data is not None:
        print("\nSample annotation:")
        print("File:", sample_path)
        print(
            json.dumps(
                sample_data,
                indent=2,
            )[:5000]
        )

    print("\nPossible split/class metadata files:")

    metadata_files = []

    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue

        name = path.name.lower()

        if any(
            word in name
            for word in (
                "train",
                "val",
                "test",
                "split",
                "class",
                "label",
            )
        ):
            metadata_files.append(path)

    for path in metadata_files[:100]:
        print(path)


if __name__ == "__main__":
    main()
