from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"


def first_existing(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


BASE_RATIO_CSV = ROOT / "data" / "lookup" / "base_ratio_table.csv"
CLASSIFIER_WEIGHTS = first_existing(
    ROOT / "weights" / "classifier" / "best.pth",
    ROOT / "weights" / "classification" / "best.pth",
)
DETECTOR_WEIGHTS = first_existing(
    ROOT / "weights" / "detector" / "best.pth",
    ROOT / "weights" / "detection" / "best.pth",
)
ADJUSTMENT_MODEL = first_existing(
    ROOT / "weights" / "adjustment" / "adjustment_model.joblib",
    ROOT / "weights" / "adjustment_model.joblib",
)

