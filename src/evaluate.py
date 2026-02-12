from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable



def extract_metrics(results) -> dict[str, float]:
    d = results.results_dict
    return {
        "precision": float(d.get("metrics/precision(B)", 0.0)),
        "recall": float(d.get("metrics/recall(B)", 0.0)),
        "map50": float(d.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(d.get("metrics/mAP50-95(B)", 0.0)),
    }


def evaluate(weights: Path, data_yaml: Path, imgsz: int, batch: int) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device="cpu",
    )
    return extract_metrics(results)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model and compute precision/recall.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args.weights, args.data_yaml, args.imgsz, args.batch)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
