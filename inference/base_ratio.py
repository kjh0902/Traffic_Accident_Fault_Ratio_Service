from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import BaseRatioResult, SceneAnchor

KEY_COLS = ("accident_place", "accident_place_feature", "vehicle_a_progress_info", "vehicle_b_progress_info")


class BaseRatioLookup:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.lookup = {tuple(str(row[col]).strip() for col in KEY_COLS): row for _, row in df.iterrows()}

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "BaseRatioLookup":
        return cls(pd.read_csv(csv_path))

    def place_options(self) -> list[str]:
        return sorted(map(str, self.df["accident_place"].dropna().unique()))

    def get(self, anchor: SceneAnchor) -> BaseRatioResult:
        key = tuple(str(getattr(anchor, col)).strip() for col in KEY_COLS)
        row = self.lookup[key]
        return BaseRatioResult(float(row["ratio_a"]), float(row["ratio_b"]), row.get("ratio_class"))

