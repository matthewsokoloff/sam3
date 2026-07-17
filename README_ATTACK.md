# Simplified SAM 3 multi-object PGD attack

This refactor replaces the original single 793-line program with a small library and one command-line runner.

## What changed

- **One model load, many images:** `--input` accepts a file or directory and processes images sequentially.
- **Many prompts:** repeat `--prompt`, or use `--road-prompts`.
- **All instances:** SAM 3 already returns every detected instance matching a prompt. The attack combines all confident masks for that prompt instead of attacking only one query.
- **More reliable:** multiple random restarts and momentum are enabled by default.
- **Easy integration:** your mentor can import `build_model()` and `run_attack()` and receive PyTorch tensors directly.
- **Simple files:** model/attack mathematics are in `sam3_attack/core.py`; file saving is in `sam3_attack/io.py`; CLI behavior is in `run_attack.py`.

## Important limitation: “segment everything”

SAM 3 is open-vocabulary and exhaustive **for a supplied concept**, but it does not automatically name every possible object in an image. To approximate road-scene “everything,” supply an ontology such as car, truck, bus, motorcycle, bicycle, person, traffic light, and traffic sign. Add or remove prompts for your experiment.

## Copy into the SAM 3 repository

Place this folder's files at the root of the cloned `facebookresearch/sam3` repository:

```text
sam3/
├── checkpoints/sam3/sam3.pt
├── sam3_attack/
├── run_attack.py
└── example_integration.py
```

## Quick commands

One image, all cars:

```bash
python run_attack.py \
  --input data/road.jpg \
  --prompt car \
  --steps 80 \
  --restarts 3
```

One image, several concepts:

```bash
python run_attack.py \
  --input data/road.jpg \
  --prompt car \
  --prompt person \
  --prompt "traffic sign"
```

A folder of road images and the built-in road ontology:

```bash
python run_attack.py \
  --input data/road_images \
  --road-prompts \
  --output outputs/road_batch
```

A stronger but slower run:

```bash
python run_attack.py \
  --input data/road.jpg \
  --road-prompts \
  --epsilon 8 \
  --step-size 1 \
  --steps 150 \
  --restarts 5
```

## How the code works

1. `build_model()` loads SAM 3, puts it in evaluation mode, and freezes its weights.
2. `build_clean_targets()` runs clean inference for every prompt and saves the union of all confident instance masks.
3. Each PGD step runs the image backbone once.
4. The grounding head runs once per prompt using the shared image features.
5. `differentiable_union()` combines all query masks. This avoids a false success where an object simply moves to another query index.
6. The loss penalizes surviving foreground, Dice overlap, query confidence, and prompt-presence confidence.
7. The input delta is updated with a momentum sign-gradient step and projected back into the requested L-infinity budget.
8. Several random restarts are attempted; the run with the smallest final objective is retained.

## Where to edit

### Add concepts

Edit `ROAD_PROMPTS` in `run_attack.py`, or pass repeated `--prompt` arguments.

### Make the attack stronger

Try these in order:

1. Increase `--restarts` from 3 to 5.
2. Increase `--steps` from 80 to 150.
3. Confirm the clean detections are good before interpreting the attack.
4. Increase `--epsilon` only after documenting the lower budgets.
5. Try several images; an image-specific attack is not expected to be equally easy on every scene.

### Change the loss

In `attack_loss_for_target()` inside `sam3_attack/core.py`:

```python
loss = foreground + dice + 0.5 * max_score + 0.25 * presence
```

- Increase `max_score` weight to focus on suppressing detection confidence.
- Increase `dice` weight to focus on destroying mask overlap.
- Remove `presence` if it dominates or behaves inconsistently.

Change one term at a time and record the configuration.

## Why the old attack was unreliable

The original script selected one clean query and then one matching query during optimization. With many objects, SAM 3 can change query ordering or distribute an object across another query. It also used only one random start. The new code attacks a differentiable union of all confident queries and uses several restarts, so it is less sensitive to query switching and unlucky initialization. The old script's one-image CLI, one selected clean query, single random initialization, and fixed 40-step loop are visible in the supplied source. 

## Research cautions

- A clean SAM 3 prediction is a pseudo-label, not ground truth.
- “Recognizable to me” is not a human-subject perceptual study.
- Report the smallest epsilon that succeeds, not only the largest.
- Keep digital proof-of-concept results separate from projected-light results.
- Process images sequentially unless the lab GPU has enough memory for true batches; backpropagation through SAM 3 is memory-intensive.
