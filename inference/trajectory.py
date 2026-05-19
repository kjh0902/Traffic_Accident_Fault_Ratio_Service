from __future__ import annotations

import math
from statistics import median

from .schemas import Evidence, Track


def speed_samples(track: Track, fps: float) -> list[float]:
    obs = track.sorted_observations()
    return [
        math.hypot(cur.center[0] - prev.center[0], cur.center[1] - prev.center[1]) * fps / max(1, cur.frame_index - prev.frame_index)
        for prev, cur in zip(obs, obs[1:])
    ]


def heading(track: Track) -> str:
    obs = track.sorted_observations()
    if len(obs) < 2:
        return "unknown"
    dx, dy = obs[-1].center[0] - obs[0].center[0], obs[-1].center[1] - obs[0].center[1]
    return ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")


def side_approach(track: Track, width: int, height: int) -> str:
    obs = track.sorted_observations()
    if not obs:
        return "unknown"
    x, y = obs[0].center
    margins = {"left": x, "right": width - x, "top": y, "bottom": height - y}
    return min(margins, key=margins.get)


def strength_from_frame_margin(margin_frames: int, fps: float) -> tuple[str | None, float]:
    sec = abs(margin_frames) / fps
    return ("strong", 0.9) if sec >= 1.5 else ("medium", 0.75) if sec >= 0.8 else ("weak", 0.55) if sec >= 0.3 else (None, 0.0)


def no_deceleration_event(track: Track, fps: float) -> dict:
    speeds = speed_samples(track, fps)
    if len(speeds) < 4:
        return {"no_deceleration": False, "no_deceleration_strength": None, "no_deceleration_confidence": 0.0}
    early, late = median(speeds[: max(2, len(speeds) // 3)]), median(speeds[-max(2, len(speeds) // 3):])
    ratio = late / early if early > 1e-6 else 0.0
    strength, conf = ("strong", 0.85) if ratio >= 0.95 else ("medium", 0.7) if ratio >= 0.8 else ("weak", 0.55) if ratio >= 0.65 else (None, 0.0)
    return {"no_deceleration": strength is not None, "no_deceleration_strength": strength, "no_deceleration_confidence": conf}


def build_evidence(tracks: list[Track], fps: float, frame_width: int, frame_height: int) -> Evidence:
    actors = {track.actor: track for track in tracks if track.actor in {"A", "B"}}
    track_a, track_b = actors.get("A"), actors.get("B")
    entry_order, first_strength, first_conf = "unknown", None, 0.0
    if track_a and track_b:
        frame_a, frame_b = track_a.sorted_observations()[0].frame_index, track_b.sorted_observations()[0].frame_index
        if frame_a != frame_b:
            entry_order = "A_first" if frame_a < frame_b else "B_first"
            first_strength, first_conf = strength_from_frame_margin(frame_a - frame_b, fps)

    rel_speed, heads, sides, events = {}, {}, {}, {}
    for actor, track in actors.items():
        speeds = speed_samples(track, fps)
        rel_speed[actor] = float(median(speeds)) if speeds else 0.0
        heads[actor] = heading(track)
        sides[actor] = side_approach(track, frame_width, frame_height)
        events[actor] = no_deceleration_event(track, fps)

    collision = None
    if track_a and track_b:
        a_last, b_last = track_a.sorted_observations()[-1], track_b.sorted_observations()[-1]
        dx, dy = a_last.center[0] - b_last.center[0], a_last.center[1] - b_last.center[1]
        collision = ("A_right_of_B" if dx > 0 else "A_left_of_B") if abs(dx) > abs(dy) else ("A_below_B" if dy > 0 else "A_above_B")

    return Evidence(
        entry_order=entry_order,
        first_entry_strength=first_strength,
        first_entry_conf=first_conf,
        relative_speed=rel_speed,
        heading=heads,
        side_approach=sides,
        collision_relative_position=collision,
        actor_events=events,
    )

