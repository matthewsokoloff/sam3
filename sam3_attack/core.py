"""Small reusable SAM 3 adapter and multi-prompt PGD attack.

The public Sam3Processor API runs under inference_mode, so this module calls
SAM 3's image backbone and grounding head directly. Model weights remain frozen;
only the input perturbation receives gradients.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

MODEL_RESOLUTION = 1008
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


@dataclass(frozen=True)
class AttackConfig:
    epsilon_pixels: float = 8.0
    step_size_pixels: float = 1.0
    steps: int = 80
    restarts: int = 3
    momentum: float = 0.9
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    amp: str = "auto"
    seed: int = 0
    early_stop_iou: float = 0.05


@dataclass
class PromptTarget:
    """Clean reference for one text prompt."""

    prompt: str
    text_features: dict[str, Any]
    clean_union_mask: torch.Tensor  # [Hm, Wm], bool
    clean_score: float
    clean_instance_count: int


@dataclass
class AttackResult:
    adversarial_image: torch.Tensor
    delta: torch.Tensor
    history: list[dict[str, float | int]]
    targets: list[PromptTarget]
    best_objective: float
    restart: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _autocast_dtype(mode: str) -> torch.dtype | None:
    if mode == "none":
        return None
    if mode == "bf16":
        _require(torch.cuda.is_bf16_supported(), "This GPU does not support bfloat16")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown AMP mode: {mode}")


def amp_context(mode: str) -> contextlib.AbstractContextManager[Any]:
    dtype = _autocast_dtype(mode)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_model(checkpoint: Path) -> torch.nn.Module:
    """Load SAM 3 once and freeze all model parameters."""
    _require(torch.cuda.is_available(), "An NVIDIA CUDA GPU is required")
    _require(checkpoint.is_file(), f"Checkpoint not found: {checkpoint}")

    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    model.eval()
    model.requires_grad_(False)
    _require(not any(p.requires_grad for p in model.parameters()), "Model was not frozen")
    return model


def make_find_stage() -> Any:
    from sam3.model.data_misc import FindStage

    return FindStage(
        img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


def prepare_text_features(model: torch.nn.Module, prompt: str, amp: str) -> dict[str, Any]:
    """Encode fixed text once; text gradients are unnecessary for an image attack."""
    _require(bool(prompt.strip()), "Prompt must not be empty")
    with torch.no_grad(), amp_context(amp):
        features = model.backbone.forward_text([prompt], device="cuda")
    return _detach_tree(features)


def preprocess_image(image_0_to_1: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of SAM 3 image preprocessing."""
    _require(image_0_to_1.ndim == 4, "Expected BCHW image tensor")
    _require(image_0_to_1.shape[:2] == (1, 3), "Expected exactly one RGB image")

    resized = F.interpolate(
        image_0_to_1.float(),
        size=(MODEL_RESOLUTION, MODEL_RESOLUTION),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (resized - IMAGE_MEAN) / IMAGE_STD


def encode_image(model: torch.nn.Module, image_0_to_1: torch.Tensor, amp: str) -> dict[str, Any]:
    """Run the expensive vision backbone once for an image."""
    model_input = preprocess_image(image_0_to_1)
    with amp_context(amp):
        return model.backbone.forward_image(model_input)


def ground_prompt(
    model: torch.nn.Module,
    image_features: dict[str, Any],
    text_features: dict[str, Any],
    find_stage: Any,
    amp: str,
) -> dict[str, torch.Tensor]:
    """Return raw masks and scores for one prompt while preserving image gradients."""
    backbone_output = dict(image_features)
    backbone_output.update(text_features)
    with amp_context(amp):
        output = model.forward_grounding(
            backbone_out=backbone_output,
            find_input=find_stage,
            find_target=None,
            geometric_prompt=model._get_dummy_prompt(),
        )

    required = {"pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec"}
    missing = required.difference(output)
    _require(not missing, f"SAM 3 raw output is missing keys: {sorted(missing)}")
    return output


def probabilities(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mask probabilities, per-query joint scores, and prompt presence."""
    masks = output["pred_masks"][0].float().sigmoid()  # [Q,Hm,Wm]
    object_scores = output["pred_logits"][0, :, 0].float().sigmoid()
    presence = output["presence_logit_dec"].reshape(-1)[0].float().sigmoid()
    joint_scores = object_scores * presence
    return masks, joint_scores, presence


def differentiable_union(mask_probs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Combine all query masks into one soft union.

    A query contributes only when both its mask probability and detection score
    are high. This prevents the attack from succeeding merely by causing SAM 3
    to move an object from one query slot to another.
    """
    weighted = (mask_probs * scores[:, None, None]).clamp(0.0, 1.0 - 1e-6)
    return 1.0 - torch.exp(torch.log1p(-weighted).sum(dim=0))


def binary_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = a.bool()
    b = b.bool()
    intersection = (a & b).sum().float()
    union = (a | b).sum().float()
    return (intersection + eps) / (union + eps)


def soft_dice(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target_f = target.float()
    intersection = (probability * target_f).sum()
    return (2.0 * intersection + eps) / (probability.sum() + target_f.sum() + eps)


def project_delta(clean: torch.Tensor, delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project into both the L-infinity ball and the legal image range."""
    delta = delta.clamp(-epsilon, epsilon)
    return torch.clamp(clean + delta, 0.0, 1.0) - clean


def build_clean_targets(
    model: torch.nn.Module,
    clean_image: torch.Tensor,
    prompts: Iterable[str],
    config: AttackConfig,
) -> list[PromptTarget]:
    """Find every confident instance for every prompt in the clean image."""
    find_stage = make_find_stage()
    text_by_prompt = {
        prompt: prepare_text_features(model, prompt, config.amp)
        for prompt in dict.fromkeys(p.strip() for p in prompts if p.strip())
    }
    _require(text_by_prompt, "At least one prompt is required")

    with torch.no_grad():
        image_features = encode_image(model, clean_image, config.amp)
        targets: list[PromptTarget] = []
        for prompt, text_features in text_by_prompt.items():
            output = ground_prompt(model, image_features, text_features, find_stage, config.amp)
            mask_probs, joint_scores, _ = probabilities(output)
            keep = joint_scores >= config.confidence_threshold
            if not bool(keep.any()):
                print(f"Skipping prompt {prompt!r}: no clean detections above threshold")
                continue

            clean_union = (mask_probs[keep] >= config.mask_threshold).any(dim=0)
            if not bool(clean_union.any()):
                print(f"Skipping prompt {prompt!r}: confident masks were empty")
                continue

            targets.append(
                PromptTarget(
                    prompt=prompt,
                    text_features=text_features,
                    clean_union_mask=clean_union,
                    clean_score=float(joint_scores[keep].max().item()),
                    clean_instance_count=int(keep.sum().item()),
                )
            )

    _require(targets, "No usable clean targets. Try clearer images, prompts, or a lower threshold.")
    return targets


def attack_loss_for_target(
    model: torch.nn.Module,
    image_features: dict[str, Any],
    target: PromptTarget,
    find_stage: Any,
    config: AttackConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure how much of one prompted concept still survives."""
    output = ground_prompt(model, image_features, target.text_features, find_stage, config.amp)
    mask_probs, joint_scores, presence = probabilities(output)
    union = differentiable_union(mask_probs, joint_scores)

    target_mask = target.clean_union_mask
    foreground = union[target_mask].mean()
    dice = soft_dice(union, target_mask)
    max_score = joint_scores.max()

    # Minimize the surviving mask, its overlap, and detection confidence.
    loss = foreground + dice + 0.5 * max_score + 0.25 * presence
    return loss, {
        "foreground": foreground,
        "dice": dice,
        "max_score": max_score,
        "presence": presence,
        "union": union,
    }


def run_attack(
    model: torch.nn.Module,
    clean_image: torch.Tensor,
    prompts: list[str],
    config: AttackConfig,
) -> AttackResult:
    """Attack every detected instance for every requested prompt.

    Multiple random restarts are the main reliability improvement over the
    original one-shot script. The best final result is returned.
    """
    targets = build_clean_targets(model, clean_image, prompts, config)
    find_stage = make_find_stage()
    epsilon = config.epsilon_pixels / 255.0
    step_size = config.step_size_pixels / 255.0

    best: AttackResult | None = None

    for restart in range(config.restarts):
        generator = torch.Generator(device="cuda").manual_seed(config.seed + restart)
        delta = torch.empty_like(clean_image).uniform_(-epsilon, epsilon, generator=generator)
        delta = project_delta(clean_image, delta, epsilon).detach().requires_grad_(True)
        velocity = torch.zeros_like(delta)
        history: list[dict[str, float | int]] = []

        for step in range(1, config.steps + 1):
            adversarial = torch.clamp(clean_image + delta, 0.0, 1.0)
            image_features = encode_image(model, adversarial, config.amp)

            # Compute one prompt loss at a time. This retains the shared image
            # backbone graph but releases each grounding-head graph immediately,
            # using much less memory than storing every prompt loss at once.
            gradient = torch.zeros_like(delta)
            metric_rows: list[dict[str, float]] = []
            objective_value = 0.0
            for index, target in enumerate(targets):
                loss, values = attack_loss_for_target(
                    model, image_features, target, find_stage, config
                )
                prompt_gradient = torch.autograd.grad(
                    loss,
                    delta,
                    only_inputs=True,
                    retain_graph=index < len(targets) - 1,
                )[0]
                gradient.add_(prompt_gradient / len(targets))
                objective_value += float(loss.detach()) / len(targets)
                metric_rows.append(
                    {
                        "foreground": float(values["foreground"].detach()),
                        "dice": float(values["dice"].detach()),
                        "max_score": float(values["max_score"].detach()),
                    }
                )

            _require(bool(torch.isfinite(gradient).all()), "Gradient contains NaN or infinity")
            _require(float(gradient.abs().max()) > 0.0, "Gradient is exactly zero")

            # Momentum makes the direction more stable across iterations.
            normalized = gradient / gradient.abs().mean().clamp_min(1e-12)
            velocity = config.momentum * velocity + normalized

            with torch.no_grad():
                delta = delta - step_size * velocity.sign()
                delta = project_delta(clean_image, delta, epsilon)
            delta = delta.detach().requires_grad_(True)

            mean_dice = sum(m["dice"] for m in metric_rows) / len(metric_rows)
            mean_fg = sum(m["foreground"] for m in metric_rows) / len(metric_rows)
            mean_score = sum(m["max_score"] for m in metric_rows) / len(metric_rows)
            row = {
                "restart": restart,
                "step": step,
                "objective": objective_value,
                "mean_foreground": mean_fg,
                "mean_dice": mean_dice,
                "mean_score": mean_score,
                "linf_pixels": float(delta.detach().abs().max() * 255.0),
            }
            history.append(row)

            if step == 1 or step % 10 == 0 or step == config.steps:
                print(
                    f"restart={restart + 1}/{config.restarts} step={step:03d}/{config.steps} "
                    f"loss={row['objective']:.4f} dice={mean_dice:.4f} "
                    f"score={mean_score:.4f} linf={row['linf_pixels']:.2f}/255"
                )

            if mean_dice <= config.early_stop_iou and mean_score < config.confidence_threshold:
                print(
			f"Early stopping at step {step}: "
			f"dice={mean_dice:.4f}, score={mean_score:.4f}"
		)
                break

        final_image = torch.clamp(clean_image + delta.detach(), 0.0, 1.0)
        final_objective = history[-1]["objective"]
        candidate = AttackResult(
            adversarial_image=final_image,
            delta=final_image - clean_image,
            history=history,
            targets=targets,
            best_objective=float(final_objective),
            restart=restart,
        )
        if best is None or candidate.best_objective < best.best_objective:
            best = candidate

        torch.cuda.empty_cache()

    assert best is not None
    return best
