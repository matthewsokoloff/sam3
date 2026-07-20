#!/usr/bin/env python3

import csv
import shutil
import zipfile
from pathlib import Path


IDS_FILE = Path(
    "data/mtsd/subset/metadata/selected_image_ids.txt"
)
DOWNLOAD_DIR = Path("data/mtsd/downloads")
OUTPUT_DIR = Path("data/mtsd/subset/images")
METADATA_DIR = Path("data/mtsd/subset/metadata")

ARCHIVES = [
    DOWNLOAD_DIR / "mtsd_fully_annotated_images.train.0.zip",
    DOWNLOAD_DIR / "mtsd_fully_annotated_images.train.1.zip",
    DOWNLOAD_DIR / "mtsd_fully_annotated_images.train.2.zip",
    DOWNLOAD_DIR / "mtsd_fully_annotated_images.val.zip",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    selected_ids = {
        line.strip()
        for line in IDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    for archive in ARCHIVES:
        if not archive.exists():
            raise FileNotFoundError(f"Missing archive: {archive}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = {
        path.stem
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

    remaining = selected_ids - existing_ids
    rows = []

    print(f"Selected image IDs: {len(selected_ids)}")
    print(f"Already extracted: {len(selected_ids & existing_ids)}")
    print(f"Remaining: {len(remaining)}")

    extracted = 0

    for archive_path in ARCHIVES:
        print(f"\nScanning {archive_path.name}", flush=True)

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if not remaining:
                    break

                if member.is_dir():
                    continue

                member_path = Path(member.filename)
                suffix = member_path.suffix.lower()

                if suffix not in IMAGE_SUFFIXES:
                    continue

                image_id = member_path.stem

                if image_id not in remaining:
                    continue

                output_path = OUTPUT_DIR / f"{image_id}{suffix}"
                temporary_path = OUTPUT_DIR / f"{image_id}{suffix}.part"

                with archive.open(member) as source:
                    with temporary_path.open("wb") as destination:
                        shutil.copyfileobj(
                            source,
                            destination,
                            length=8 * 1024 * 1024,
                        )

                temporary_path.replace(output_path)

                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": str(output_path.resolve()),
                        "source_archive": archive_path.name,
                        "source_member": member.filename,
                    }
                )

                remaining.remove(image_id)
                extracted += 1

                if extracted % 100 == 0:
                    print(
                        f"Extracted {extracted}; "
                        f"remaining {len(remaining)}",
                        flush=True,
                    )

        print(
            f"Finished {archive_path.name}; "
            f"remaining {len(remaining)}",
            flush=True,
        )

    index_path = METADATA_DIR / "image_index.csv"

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "image_path",
                "source_archive",
                "source_member",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    missing_path = METADATA_DIR / "missing_image_ids.txt"
    missing_path.write_text(
        "".join(f"{image_id}\n" for image_id in sorted(remaining)),
        encoding="utf-8",
    )

    final_ids = {
        path.stem
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

    found = selected_ids & final_ids

    print("\nFinished")
    print(f"Selected: {len(selected_ids)}")
    print(f"Found: {len(found)}")
    print(f"Missing: {len(remaining)}")
    print(f"Images: {OUTPUT_DIR.resolve()}")
    print(f"Index: {index_path.resolve()}")

    if remaining:
        raise SystemExit(
            "Some images were not found. "
            "See missing_image_ids.txt."
        )


if __name__ == "__main__":
    main()
