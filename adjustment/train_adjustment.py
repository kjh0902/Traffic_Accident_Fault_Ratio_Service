from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUSTMENT_ROOT = PROJECT_ROOT / "table"
DEFAULT_INPUT_CSV = ADJUSTMENT_ROOT / "adjustment_outputs" / "adjustment_input.csv"
DEFAULT_OUTPUT_DIR = ADJUSTMENT_ROOT / "adjustment_outputs"

STRENGTH_LEVELS = ("weak", "medium", "strong")
REQUIRED_EVIDENCE_COLS = (
    "entry_order",
    "first_entry_strength",
    "first_entry_conf",
    "A_no_deceleration",
    "A_no_deceleration_strength",
    "A_no_deceleration_conf",
    "B_no_deceleration",
    "B_no_deceleration_strength",
    "B_no_deceleration_conf",
)

def safe_strength(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value if value in STRENGTH_LEVELS else None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def evidence_to_design_row(row: dict[str, Any]) -> dict[str, float]:
    feat: dict[str, float] = {}

    entry_order = row.get("entry_order")
    entry_strength = safe_strength(row.get("first_entry_strength"))
    entry_conf = float(row.get("first_entry_conf", 0.0) or 0.0)
    for who in ("A_first", "B_first"):
        for strength in STRENGTH_LEVELS:
            feat[f"first_entry_{who}_{strength}"] = 0.0
    if entry_order in ("A_first", "B_first") and entry_strength is not None:
        feat[f"first_entry_{entry_order}_{entry_strength}"] = entry_conf

    for actor in ("A", "B"):
        for strength in STRENGTH_LEVELS:
            feat[f"{actor}_no_decel_{strength}"] = 0.0
            feat[f"{actor}_evasive_{strength}"] = 0.0

        no_decel_strength = safe_strength(row.get(f"{actor}_no_deceleration_strength"))
        no_decel_conf = float(row.get(f"{actor}_no_deceleration_conf", 0.0) or 0.0)
        if as_bool(row.get(f"{actor}_no_deceleration", False)) and no_decel_strength is not None:
            feat[f"{actor}_no_decel_{no_decel_strength}"] = no_decel_conf

        evasive_strength = safe_strength(row.get(f"{actor}_evasive_action_strength"))
        evasive_conf = float(row.get(f"{actor}_evasive_action_conf", 0.0) or 0.0)
        if as_bool(row.get(f"{actor}_evasive_action", False)) and evasive_strength is not None:
            feat[f"{actor}_evasive_{evasive_strength}"] = evasive_conf

    return feat


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = [evidence_to_design_row(row.to_dict()) for _, row in df.iterrows()]
    x = pd.DataFrame(rows).fillna(0.0)
    return x, list(x.columns)


def build_learned_adjustment_table(feature_cols: list[str], weights) -> pd.DataFrame:
    rows = []
    for feature, weight in zip(feature_cols, weights, strict=False):
        parts = feature.split("_")
        if feature.startswith("first_entry_"):
            group = "first_entry"
            key = "_".join(parts[2:-1])
            level = parts[-1]
        else:
            group = parts[1] if len(parts) >= 3 else "unknown"
            key = parts[0]
            level = parts[-1]
        rows.append(
            {
                "feature": feature,
                "group": group,
                "key": key,
                "level": level,
                "weight": float(weight),
                "abs_weight": abs(float(weight)),
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_weight", "feature"], ascending=[False, True])


def train(input_csv: Path) -> tuple[dict[str, Any], dict[str, float]]:
    df = pd.read_csv(input_csv)
    for col in ("base_ratio_a", "true_ratio_a"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    missing_evidence = [col for col in REQUIRED_EVIDENCE_COLS if col not in df.columns]
    if missing_evidence:
        raise ValueError(
            "Adjustment CSV does not contain tracking evidence columns. "
            f"Run build_adjustment_csv.py first. Missing: {missing_evidence}"
        )

    df = df.copy()
    df["target_adjustment"] = df["true_ratio_a"].astype(float) - df["base_ratio_a"].astype(float)
    x, feature_cols = build_design_matrix(df)
    if not feature_cols:
        raise ValueError("No adjustment features were generated.")
    y = df["target_adjustment"].astype(float)

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 30.0])
    metrics: dict[str, float] = {
        "num_rows": float(len(df)),
        "num_features": float(len(feature_cols)),
        "target_mean": float(y.mean()),
        "target_mae_if_zero": float(y.abs().mean()),
    }
    if len(df) >= 5:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        metrics["val_mae"] = float(mean_absolute_error(y_val, pred))
    else:
        model.fit(x, y)
        pred = model.predict(x)
        metrics["train_mae"] = float(mean_absolute_error(y, pred))

    table = build_learned_adjustment_table(feature_cols, model.coef_)
    payload = {"model": model, "feature_cols": feature_cols, "table": table}
    return payload, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adjustment_model.joblib from adjustment CSV.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, metrics = train(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "adjustment_model.joblib"
    table_path = args.output_dir / "learned_adjustment_table.csv"
    metrics_path = args.output_dir / "adjustment_metrics.json"

    joblib.dump(payload, model_path)
    payload["table"].to_csv(table_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"saved: {model_path}")
    print(f"saved: {table_path}")
    print(f"saved: {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
