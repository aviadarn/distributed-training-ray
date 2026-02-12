from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _train_worker(config: dict) -> None:
    from ray import train
    from ultralytics import YOLO

    ctx = train.get_context()
    rank = ctx.get_world_rank()

    # We run training on rank 0 to keep this example stable on CPU-only setups.
    if rank == 0:
        model = YOLO(config["model"])
        results = model.train(
            data=config["data_yaml"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            project=config["project_dir"],
            name=config["run_name"],
            workers=0,
            device="cpu",
            verbose=True,
        )

        metrics = {
            "fitness": float(results.results_dict.get("fitness", 0.0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        }
        train.report(metrics)
    else:
        train.report({"worker": rank, "status": "idle"})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed YOLO12 training with Ray.")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--project-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-name", type=str, default="ray-yolo12-smoke")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    import ray
    args.project_dir.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=_train_worker,
        train_loop_config={
            "data_yaml": str(args.data_yaml),
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project_dir": str(args.project_dir),
            "run_name": args.run_name,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=False,
        ),
        run_config=RunConfig(name=f"trainer-{args.run_name}"),
    )

    result = trainer.fit()
    print("Ray training finished")
    print(result.metrics)


if __name__ == "__main__":
    main()
