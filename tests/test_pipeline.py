from pathlib import Path

from src.data import create_dataset
from src.evaluate import extract_metrics


class _DummyResults:
    def __init__(self):
        self.results_dict = {
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.75,
            "metrics/mAP50(B)": 0.77,
            "metrics/mAP50-95(B)": 0.44,
        }


def test_dataset_generation(tmp_path: Path) -> None:
    data_yaml = create_dataset(tmp_path / "small", train_images=4, val_images=2, img_size=64)

    assert data_yaml.exists()
    train_images = list((tmp_path / "small" / "images" / "train").glob("*.jpg"))
    train_labels = list((tmp_path / "small" / "labels" / "train").glob("*.txt"))

    assert len(train_images) == 4
    assert len(train_labels) == 4


def test_metric_extraction() -> None:
    metrics = extract_metrics(_DummyResults())

    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.75
    assert metrics["map50"] == 0.77
    assert metrics["map50_95"] == 0.44
