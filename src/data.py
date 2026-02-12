from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable


CLASSES = ["rectangle"]


def _ensure_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _normalized_bbox(x1: int, y1: int, x2: int, y2: int, img_size: int) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2) / img_size
    cy = ((y1 + y2) / 2) / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return cx, cy, w, h


def _write_image_with_pillow(image_path: Path, img_size: int, x1: int, y1: int, x2: int, y2: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (img_size, img_size), color=(15, 15, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 50), width=2, fill=(80, 120, 240))
    image.save(image_path)


def _write_placeholder_image(image_path: Path) -> None:
    # Fallback for environments without Pillow. Real training requires valid images;
    # this fallback keeps smoke tests functional.
    image_path.write_bytes(b"\xff\xd8\xff\xd9")


def _write_example(image_path: Path, label_path: Path, img_size: int, rng: random.Random) -> None:
    x1 = rng.randint(8, img_size // 2)
    y1 = rng.randint(8, img_size // 2)
    x2 = rng.randint(img_size // 2, img_size - 8)
    y2 = rng.randint(img_size // 2, img_size - 8)

    try:
        _write_image_with_pillow(image_path=image_path, img_size=img_size, x1=x1, y1=y1, x2=x2, y2=y2)
    except ModuleNotFoundError:
        _write_placeholder_image(image_path)

    cx, cy, w, h = _normalized_bbox(x1, y1, x2, y2, img_size)
    with label_path.open("w", encoding="utf-8") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _generate_split(root: Path, split: str, count: int, img_size: int, seed: int) -> None:
    rng = random.Random(seed)
    for idx in range(count):
        stem = f"{split}_{idx:04d}"
        image_path = root / "images" / split / f"{stem}.jpg"
        label_path = root / "labels" / split / f"{stem}.txt"
        _write_example(image_path=image_path, label_path=label_path, img_size=img_size, rng=rng)


def write_data_yaml(root: Path) -> Path:
    data_yaml_path = root / "data.yaml"
    content = "\n".join(
        [
            f"path: {root.resolve()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: rectangle",
            "",
        ]
    )
    data_yaml_path.write_text(content, encoding="utf-8")
    return data_yaml_path


def create_dataset(output_dir: Path, train_images: int, val_images: int, img_size: int, seed: int = 42) -> Path:
    _ensure_dirs(output_dir)
    _generate_split(output_dir, "train", train_images, img_size, seed)
    _generate_split(output_dir, "val", val_images, img_size, seed + 1)
    return write_data_yaml(output_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny YOLO detection dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/small"))
    parser.add_argument("--train-images", type=int, default=24)
    parser.add_argument("--val-images", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    data_yaml = create_dataset(
        output_dir=args.output_dir,
        train_images=args.train_images,
        val_images=args.val_images,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"Dataset created at: {args.output_dir}")
    print(f"Data config: {data_yaml}")


if __name__ == "__main__":
    main()
