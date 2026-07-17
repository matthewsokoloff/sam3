"""Input/output helpers kept separate from the attack mathematics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import AttackResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(path: Path) -> list[Path]:
    """Accept either one image or a directory of images."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(path)


def load_image(path: Path) -> tuple[torch.Tensor, Image.Image]:
    pil = Image.open(path).convert("RGB")
    array = np.asarray(pil, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda")
    return tensor, pil


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(np.rint(array * 255).astype(np.uint8))


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    return Image.fromarray(mask.detach().bool().cpu().numpy().astype(np.uint8) * 255)


def overlay(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    base = image.convert("RGBA")
    alpha = mask_to_pil(mask).point(lambda value: 110 if value else 0)
    layer = Image.new("RGBA", base.size, (255, 40, 40, 0))
    layer.putalpha(alpha.resize(base.size, Image.Resampling.NEAREST))
    return Image.alpha_composite(base, layer).convert("RGB")


def save_result(
    output_dir: Path,
    clean_pil: Image.Image,
    result: AttackResult,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adversarial_pil = tensor_to_pil(result.adversarial_image)
    adversarial_pil.save(output_dir / "adversarial.png")

    # Delta visualization is centered around gray; actual delta is saved as NPY.
    epsilon = max(float(result.delta.abs().max()), 1e-8)
    delta_view = tensor_to_pil((0.5 + result.delta / (2 * epsilon)).clamp(0, 1))
    delta_view.save(output_dir / "delta_visualization.png")
    np.save(output_dir / "delta.npy", result.delta.float().cpu().numpy())

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)

    metadata = {
        **metadata,
        "best_restart": result.restart,
        "best_objective": result.best_objective,
        "delta_linf_pixels": float(result.delta.abs().max() * 255.0),
        "targets": [
            {
                "prompt": t.prompt,
                "clean_score": t.clean_score,
                "clean_instance_count": t.clean_instance_count,
            }
            for t in result.targets
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Simple side-by-side image.
    width = min(600, clean_pil.width)
    height = round(clean_pil.height * width / clean_pil.width)
    left = clean_pil.resize((width, height))
    right = adversarial_pil.resize((width, height))
    canvas = Image.new("RGB", (width * 2, height + 30), "white")
    canvas.paste(left, (0, 30))
    canvas.paste(right, (width, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Clean", fill="black")
    draw.text((width + 8, 8), "Adversarial", fill="black")
    canvas.save(output_dir / "comparison.png")
