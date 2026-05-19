from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .schemas import AdjustmentResult, Evidence

STRENGTH_LEVELS = ("weak", "medium", "strong")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return False if value is None or pd.isna(value) else str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_strength(value: Any) -> str | None:
    return value if value in STRENGTH_LEVELS else None


def evidence_to_flat_dict(evidence: Evidence) -> dict[str, Any]:
    data = asdict(evidence)
    events, speeds, heads, sides = data.pop("actor_events", {}), data.pop("relative_speed", {}), data.pop("heading", {}), data.pop("side_approach", {})
    for actor in ("A", "B"):
        data[f"{actor}_relative_speed"] = speeds.get(actor, 0.0)
        data[f"{actor}_heading"] = heads.get(actor, "unknown")
        data[f"{actor}_side_approach"] = sides.get(actor, "unknown")
        for key in ("no_deceleration", "evasive_action"):
            event = events.get(actor, {})
            data[f"{actor}_{key}"] = event.get(key, False)
            data[f"{actor}_{key}_strength"] = event.get(f"{key}_strength")
            data[f"{actor}_{key}_conf"] = event.get(f"{key}_confidence", 0.0)
    return data


def design_row(evidence: Evidence) -> dict[str, float]:
    row, feat = evidence_to_flat_dict(evidence), {}
    for who in ("A_first", "B_first"):
        for strength in STRENGTH_LEVELS:
            feat[f"first_entry_{who}_{strength}"] = 0.0
    strength = safe_strength(row.get("first_entry_strength"))
    if row.get("entry_order") in ("A_first", "B_first") and strength:
        feat[f"first_entry_{row['entry_order']}_{strength}"] = float(row.get("first_entry_conf", 0.0) or 0.0)

    for actor in ("A", "B"):
        for event, short in (("no_deceleration", "no_decel"), ("evasive_action", "evasive")):
            for strength in STRENGTH_LEVELS:
                feat[f"{actor}_{short}_{strength}"] = 0.0
            strength = safe_strength(row.get(f"{actor}_{event}_strength"))
            if as_bool(row.get(f"{actor}_{event}")) and strength:
                feat[f"{actor}_{short}_{strength}"] = float(row.get(f"{actor}_{event}_conf", 0.0) or 0.0)
    return feat


class AdjustmentModel:
    def __init__(self, payload: dict[str, Any] | None = None):
        self.payload = payload or {"model": None, "feature_cols": []}

    @classmethod
    def load(cls, path: str | Path) -> "AdjustmentModel":
        return cls(joblib.load(path))

    @classmethod
    def zero(cls) -> "AdjustmentModel":
        return cls()

    def predict_adjustment(self, evidence: Evidence) -> float:
        model = self.payload.get("model")
        if model is None:
            return 0.0
        row = design_row(evidence)
        x = pd.DataFrame([{col: row.get(col, 0.0) for col in self.payload["feature_cols"]}])
        return float(model.predict(x)[0])

    def apply(self, base_ratio_a: float, evidence: Evidence, round_to: int | None = 10) -> AdjustmentResult:
        adjustment = self.predict_adjustment(evidence)
        ratio_a = max(0.0, min(100.0, base_ratio_a + adjustment))
        if round_to:
            ratio_a = round(ratio_a / round_to) * round_to
        return AdjustmentResult(adjustment, ratio_a, 100.0 - ratio_a)

