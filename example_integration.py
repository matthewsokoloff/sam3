"""Minimal example showing how another research program can call the attack."""

from pathlib import Path

from sam3_attack import AttackConfig, build_model, run_attack
from sam3_attack.io import load_image

model = build_model(Path("checkpoints/sam3/sam3.pt"))
image, _ = load_image(Path("data/road.jpg"))

config = AttackConfig(steps=80, restarts=3, epsilon_pixels=8)
result = run_attack(model, image, prompts=["car", "person"], config=config)

# Your mentor can use these tensors directly without reading saved files.
adversarial_image = result.adversarial_image  # [1,3,H,W], values in [0,1]
delta = result.delta                          # same shape, signed perturbation
