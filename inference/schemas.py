from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    frame_index: int
    bbox_xyxy: Box
    score: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass(frozen=True)
class TrackedObject:
    frame_index: int
    track_id: int
    bbox_xyxy: Box
    score: float
    label: str
    actor: Literal["A", "B"] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class Track:
    track_id: int
    label: str
    actor: Literal["A", "B"] | None
    observations: list[TrackedObject] = field(default_factory=list)

    def sorted_observations(self) -> list[TrackedObject]:
        return sorted(self.observations, key=lambda x: x.frame_index)


@dataclass(frozen=True)
class SceneAnchor:
    accident_place: str
    accident_place_feature: str
    vehicle_a_progress_info: str
    vehicle_b_progress_info: str
    confidence: float = 0.0


@dataclass(frozen=True)
class Evidence:
    entry_order: Literal["A_first", "B_first", "unknown"] = "unknown"
    first_entry_strength: Literal["weak", "medium", "strong"] | None = None
    first_entry_conf: float = 0.0
    relative_speed: dict[str, float] = field(default_factory=dict)
    heading: dict[str, str] = field(default_factory=dict)
    side_approach: dict[str, str] = field(default_factory=dict)
    collision_relative_position: str | None = None
    actor_events: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseRatioResult:
    ratio_a: float
    ratio_b: float
    ratio_class: str | int | None = None


@dataclass(frozen=True)
class AdjustmentResult:
    adjustment_a: float
    ratio_a: float
    ratio_b: float

