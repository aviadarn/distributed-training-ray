# Ray Distributed YOLO12 Object Detection Example

This repository contains an end-to-end **example project** for distributed object-detection training with **Ray** and **Ultralytics YOLO12**. It includes:

- A synthetic small-image dataset generator in YOLO format.
- A Ray-powered training entrypoint for distributed orchestration.
- An evaluation pipeline that computes and stores **precision** and **recall**.
- A Docker image to run the full workflow in a reproducible environment.
- Lightweight tests for smoke validation on a small amount of data.

## Architecture Overview

1. `src/data.py` creates a tiny synthetic dataset (colored rectangles on blank backgrounds) in YOLO detection format.
2. `src/train_ray_yolo.py` starts Ray and launches a distributed training job with `TorchTrainer`.
3. The worker training function uses Ultralytics YOLO with model `yolo12n.pt` (configurable).
4. `src/evaluate.py` runs validation and extracts precision/recall metrics from YOLO results.
5. Metrics are saved to `artifacts/metrics.json`.

> Note: This project is designed as a practical example. For production, expand data ingestion, experiment tracking, and distributed strategy (e.g. multi-GPU, checkpoint sync, model registry).

---

## Quickstart (Local)

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Generate a tiny dataset

```bash
python -m src.data --output-dir data/small --train-images 24 --val-images 8 --img-size 96
```

### 3) Run distributed training with Ray

```bash
python -m src.train_ray_yolo \
  --data-yaml data/small/data.yaml \
  --model yolo12n.pt \
  --epochs 1 \
  --imgsz 96 \
  --batch 8 \
  --num-workers 2 \
  --run-name ray-yolo12-smoke
```

### 4) Evaluate precision & recall

```bash
python -m src.evaluate \
  --weights artifacts/ray-yolo12-smoke/weights/best.pt \
  --data-yaml data/small/data.yaml \
  --imgsz 96 \
  --batch 8 \
  --output artifacts/metrics.json
```

Example output:

```json
{
  "precision": 0.72,
  "recall": 0.69,
  "map50": 0.70,
  "map50_95": 0.41
}
```

---

## Docker Usage

### Build image

```bash
docker build -t ray-yolo12-example:latest .
```

### Run full smoke workflow

```bash
docker run --rm -it ray-yolo12-example:latest
```

The container command performs:

1. Tiny dataset generation.
2. 1-epoch Ray orchestration training.
3. Precision/recall evaluation.
4. Prints final metrics.

---

## Distributed Training Notes

- Ray coordinates workers and resource scheduling via `ScalingConfig`.
- The script is intentionally CPU-friendly for smoke testing.
- For real distributed speedup, attach GPUs and increase workers, dataset size, and epochs.

---

## Testing (small data)

Run:

```bash
pytest -q
```

Tests validate:

- Dataset generation format and files.
- Metric extraction and serialization.

---

## File layout

- `src/data.py` — synthetic YOLO dataset creation.
- `src/train_ray_yolo.py` — Ray distributed training entrypoint.
- `src/evaluate.py` — precision/recall evaluation.
- `tests/test_pipeline.py` — small-data checks.
- `Dockerfile` — reproducible environment.

