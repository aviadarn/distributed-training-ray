#!/usr/bin/env bash
set -euo pipefail

python -m src.data --output-dir data/small --train-images 16 --val-images 4 --img-size 96
python -m src.train_ray_yolo \
  --data-yaml data/small/data.yaml \
  --model yolo12n.pt \
  --epochs 1 \
  --imgsz 96 \
  --batch 8 \
  --num-workers 2 \
  --run-name ray-yolo12-smoke
python -m src.evaluate \
  --weights artifacts/ray-yolo12-smoke/weights/best.pt \
  --data-yaml data/small/data.yaml \
  --imgsz 96 \
  --batch 8 \
  --output artifacts/metrics.json
cat artifacts/metrics.json
