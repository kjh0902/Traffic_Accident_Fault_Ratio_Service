from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUSTMENT_ROOT = PROJECT_ROOT / "table"
PROCESSED_ROOT = ADJUSTMENT_ROOT / "adjustment_data" / "processed"
OUTPUT_ROOT = ADJUSTMENT_ROOT / "adjustment_outputs"

DEFAULT_INPUT_JSON = PROCESSED_ROOT / "adjustment_input.json"
DEFAULT_CLASS_MAPS = PROCESSED_ROOT / "class_maps.json"
DEFAULT_BASE_RATIO_CSV = ADJUSTMENT_ROOT / "adjustment_data" / "base_ratio_table.csv"
DEFAULT_DETECTOR_WEIGHTS = ADJUSTMENT_ROOT / "detect" / "best.pth"
DEFAULT_OUTPUT_CSV = OUTPUT_ROOT / "adjustment_input.csv"

KEY_COLS = (
    "accident_place",
    "accident_place_feature",
    "vehicle_a_progress_info",
    "vehicle_b_progress_info",
)


@dataclass(frozen=True)
class Detection:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackedObject:
    frame_index: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str
    actor: Literal["A", "B"] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    track_id: int
    label: str
    actor: Literal["A", "B"] | None
    observations: list[TrackedObject] = field(default_factory=list)

    def sorted_observations(self) -> list[TrackedObject]:
        return sorted(self.observations, key=lambda x: x.frame_index)


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


class FasterRCNNDetector:
    def __init__(self, weights_path: Path, device: str, score_threshold: float = 0.5):
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        ckpt = torch.load(weights_path, map_location=self.device)
        self.classes = tuple(ckpt["classes"])
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            int(ckpt["num_classes_with_background"]),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    def label_name(self, contiguous_idx: int) -> str:
        return self.classes[contiguous_idx - 1]

    @torch.inference_mode()
    def detect_frame(self, frame, frame_index: int) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).to(self.device)
        output = self.model([tensor])[0]
        detections = []
        for box, score, label_idx in zip(output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()):
            if float(score.item()) < self.score_threshold:
                continue
            raw_label = self.label_name(int(label_idx.item()))
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    frame_index=frame_index,
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=float(score.item()),
                    label=raw_label,
                )
            )
        return detections

    def detect_video(self, video_path: Path, frame_stride: int = 1) -> tuple[list[list[Detection]], float, int, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detections_by_frame = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                detections_by_frame.append(self.detect_frame(frame, frame_index))
            else:
                detections_by_frame.append([])
            frame_index += 1
        cap.release()
        return detections_by_frame, float(fps), width, height


def iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, min_track_length: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_track_length = min_track_length

    def track(self, detections_by_frame: list[list[Detection]]) -> list[TrackedObject]:
        active = []
        finished = []
        next_id = 1
        for frame_dets in detections_by_frame:
            used = set()
            for track in active:
                best_id, best_score = None, 0.0
                for det_id, det in enumerate(frame_dets):
                    if det_id in used or det.label != track["label"]:
                        continue
                    score = iou(track["bbox"], det.bbox_xyxy)
                    if score > best_score:
                        best_id, best_score = det_id, score
                if best_id is not None and best_score >= self.iou_threshold:
                    det = frame_dets[best_id]
                    track["bbox"] = det.bbox_xyxy
                    track["missed"] = 0
                    track["history"].append(det)
                    used.add(best_id)
                else:
                    track["missed"] += 1
            alive = []
            for track in active:
                if track["missed"] > self.max_missed:
                    finished.append(track)
                else:
                    alive.append(track)
            active = alive
            for det_id, det in enumerate(frame_dets):
                if det_id in used:
                    continue
                active.append({"track_id": next_id, "label": det.label, "bbox": det.bbox_xyxy, "missed": 0, "history": [det]})
                next_id += 1

        objects = []
        for track in finished + active:
            if len(track["history"]) < self.min_track_length:
                continue
            for det in track["history"]:
                objects.append(
                    TrackedObject(
                        frame_index=det.frame_index,
                        track_id=track["track_id"],
                        bbox_xyxy=det.bbox_xyxy,
                        score=det.score,
                        label=det.label,
                    )
                )
        return objects


def tracks_from_objects(objects: list[TrackedObject]) -> list[Track]:
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj.track_id].append(obj)
    tracks = []
    for track_id, observations in grouped.items():
        observations = sorted(observations, key=lambda x: x.frame_index)
        actor = next((obs.actor for obs in observations if obs.actor is not None), None)
        tracks.append(Track(track_id=track_id, label=observations[0].label, actor=actor, observations=observations))
    return tracks


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def assign_actors_by_size(objects: list[TrackedObject]) -> list[TrackedObject]:
    first_obs = {}
    for obj in sorted(objects, key=lambda x: x.frame_index):
        first_obs.setdefault(obj.track_id, obj)
    ranked = sorted(first_obs.values(), key=lambda x: box_area(x.bbox_xyxy), reverse=True)
    actor_map = {obs.track_id: actor for obs, actor in zip(ranked[:2], ("A", "B"))}
    return [
        TrackedObject(o.frame_index, o.track_id, o.bbox_xyxy, o.score, o.label, actor_map.get(o.track_id))
        for o in objects
    ]


def assign_actors_from_annotation(objects: list[TrackedObject], annotation_path: Path, min_iou: float = 0.2) -> list[TrackedObject]:
    with annotation_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    track_to_actor: dict[int, str] = {}
    for actor in ("A", "B"):
        ann = annotations.get(actor)
        if not ann:
            continue
        ann_frame = int(ann.get("frame_index", 0))
        ann_box = tuple(float(v) for v in ann["bbox_xyxy"])
        best_track, best_iou = None, 0.0
        for obj in objects:
            if obj.frame_index != ann_frame:
                continue
            score = iou(obj.bbox_xyxy, ann_box)
            if score > best_iou:
                best_track, best_iou = obj.track_id, score
        if best_track is not None and best_iou >= min_iou:
            track_to_actor[best_track] = actor
    return [
        TrackedObject(o.frame_index, o.track_id, o.bbox_xyxy, o.score, o.label, track_to_actor.get(o.track_id, o.actor))
        for o in objects
    ]


def speed_samples(track: Track, fps: float) -> list[float]:
    obs = track.sorted_observations()
    speeds = []
    for prev, cur in zip(obs, obs[1:], strict=False):
        dt_frames = max(1, cur.frame_index - prev.frame_index)
        speeds.append(math.hypot(cur.center[0] - prev.center[0], cur.center[1] - prev.center[1]) * fps / dt_frames)
    return speeds


def heading(track: Track) -> str:
    obs = track.sorted_observations()
    if len(obs) < 2:
        return "unknown"
    dx = obs[-1].center[0] - obs[0].center[0]
    dy = obs[-1].center[1] - obs[0].center[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def strength_from_frame_margin(margin_frames: int, fps: float) -> tuple[str | None, float]:
    margin_sec = abs(margin_frames) / fps
    if margin_sec >= 1.5:
        return "strong", 0.9
    if margin_sec >= 0.8:
        return "medium", 0.75
    if margin_sec >= 0.3:
        return "weak", 0.55
    return None, 0.0


def no_deceleration_event(track: Track, fps: float) -> dict:
    speeds = speed_samples(track, fps)
    if len(speeds) < 4:
        return {"no_deceleration": False, "no_deceleration_strength": None, "no_deceleration_confidence": 0.0}
    early = median(speeds[: max(2, len(speeds) // 3)])
    late = median(speeds[-max(2, len(speeds) // 3) :])
    ratio = late / early if early > 1e-6 else 0.0
    if ratio >= 0.95:
        strength, conf = "strong", 0.85
    elif ratio >= 0.8:
        strength, conf = "medium", 0.7
    elif ratio >= 0.65:
        strength, conf = "weak", 0.55
    else:
        strength, conf = None, 0.0
    return {"no_deceleration": strength is not None, "no_deceleration_strength": strength, "no_deceleration_confidence": conf}


def side_approach(track: Track, frame_width: int, frame_height: int) -> str:
    obs = track.sorted_observations()
    if not obs:
        return "unknown"
    x, y = obs[0].center
    margins = {"left": x, "right": frame_width - x, "top": y, "bottom": frame_height - y}
    return min(margins, key=margins.get)


def build_evidence(tracks: list[Track], fps: float, frame_width: int, frame_height: int) -> Evidence:
    actors = {track.actor: track for track in tracks if track.actor in {"A", "B"}}
    track_a = actors.get("A")
    track_b = actors.get("B")
    entry_order = "unknown"
    first_entry_strength = None
    first_entry_conf = 0.0
    if track_a and track_b:
        entry_a = track_a.sorted_observations()[0].frame_index
        entry_b = track_b.sorted_observations()[0].frame_index
        if entry_a != entry_b:
            entry_order = "A_first" if entry_a < entry_b else "B_first"
            first_entry_strength, first_entry_conf = strength_from_frame_margin(entry_a - entry_b, fps)

    actor_events = {}
    relative_speed = {}
    heading_map = {}
    side_map = {}
    for actor, track in actors.items():
        speeds = speed_samples(track, fps)
        relative_speed[actor] = float(median(speeds)) if speeds else 0.0
        heading_map[actor] = heading(track)
        side_map[actor] = side_approach(track, frame_width, frame_height)
        actor_events[actor] = no_deceleration_event(track, fps)

    collision_relative_position = None
    if track_a and track_b:
        a_last = track_a.sorted_observations()[-1]
        b_last = track_b.sorted_observations()[-1]
        dx = a_last.center[0] - b_last.center[0]
        dy = a_last.center[1] - b_last.center[1]
        if abs(dx) > abs(dy):
            collision_relative_position = "A_right_of_B" if dx > 0 else "A_left_of_B"
        else:
            collision_relative_position = "A_below_B" if dy > 0 else "A_above_B"

    return Evidence(
        entry_order=entry_order,  # type: ignore[arg-type]
        first_entry_strength=first_entry_strength,  # type: ignore[arg-type]
        first_entry_conf=first_entry_conf,
        relative_speed=relative_speed,
        heading=heading_map,
        side_approach=side_map,
        collision_relative_position=collision_relative_position,
        actor_events=actor_events,
    )


def evidence_to_flat_dict(evidence: Evidence) -> dict[str, Any]:
    data = asdict(evidence)
    actor_events = data.pop("actor_events", {})
    relative_speed = data.pop("relative_speed", {})
    heading_map = data.pop("heading", {})
    side_map = data.pop("side_approach", {})
    flat = data
    for actor in ("A", "B"):
        flat[f"{actor}_relative_speed"] = relative_speed.get(actor, 0.0)
        flat[f"{actor}_heading"] = heading_map.get(actor, "unknown")
        flat[f"{actor}_side_approach"] = side_map.get(actor, "unknown")
        events = actor_events.get(actor, {})
        flat[f"{actor}_no_deceleration"] = events.get("no_deceleration", False)
        flat[f"{actor}_no_deceleration_strength"] = events.get("no_deceleration_strength")
        flat[f"{actor}_no_deceleration_conf"] = events.get("no_deceleration_confidence", 0.0)
        flat[f"{actor}_evasive_action"] = events.get("evasive_action", False)
        flat[f"{actor}_evasive_action_strength"] = events.get("evasive_action_strength")
        flat[f"{actor}_evasive_action_conf"] = events.get("evasive_action_confidence", 0.0)
    return flat


class BaseRatioLookup:
    def __init__(self, csv_path: Path):
        rows = pd.read_csv(csv_path)
        self.lookup = {tuple(str(row[col]).strip() for col in KEY_COLS): row.to_dict() for _, row in rows.iterrows()}

    def get(self, accident_place: str, accident_place_feature: str, vehicle_a_progress_info: str, vehicle_b_progress_info: str) -> float:
        key = tuple(str(v).strip() for v in (accident_place, accident_place_feature, vehicle_a_progress_info, vehicle_b_progress_info))
        row = self.lookup.get(key)
        if row is None:
            raise KeyError(key)
        return float(row["ratio_a"])


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_prediction(key: str, value: str, class_maps: dict[str, dict[str, str]]) -> str:
    return class_maps.get(key, {}).get(str(value), str(value))


def load_label_anchor(item: dict, class_maps: dict[str, dict[str, str]]) -> tuple[str, str, str, str]:
    label_path = item.get("label_path")
    if not label_path:
        raise ValueError("label anchor requires label_path in adjustment_input.json")
    data = load_json(Path(label_path))
    video = data.get("video", data)
    return (
        decode_prediction("accident_place", str(video["accident_place"]), class_maps),
        decode_prediction("accident_place_feature", str(video["accident_place_feature"]), class_maps),
        decode_prediction("vehicle_a_progress_info", str(video["vehicle_a_progress_info"]), class_maps),
        decode_prediction("vehicle_b_progress_info", str(video["vehicle_b_progress_info"]), class_maps),
    )


def process_item(
    item: dict,
    detector: FasterRCNNDetector,
    tracker: IoUTracker,
    base_lookup: BaseRatioLookup,
    class_maps: dict[str, dict[str, str]],
    frame_stride: int,
) -> dict | None:
    video_path = Path(item["video_path"])
    accident_place, accident_place_feature, vehicle_a, vehicle_b = load_label_anchor(item, class_maps)

    base_ratio_a = base_lookup.get(accident_place, accident_place_feature, vehicle_a, vehicle_b)
    detections_by_frame, fps, width, height = detector.detect_video(video_path, frame_stride=frame_stride)
    objects = tracker.track(detections_by_frame)
    annotation_path = item.get("annotation_path")
    if annotation_path and Path(annotation_path).exists():
        objects = assign_actors_from_annotation(objects, Path(annotation_path))
    else:
        objects = assign_actors_by_size(objects)
    evidence = build_evidence(tracks_from_objects(objects), fps=fps, frame_width=width, frame_height=height)

    row = {
        "video_path": str(video_path),
        "label_path": item.get("label_path"),
        "accident_place": accident_place,
        "accident_place_feature": accident_place_feature,
        "vehicle_a_progress_info": vehicle_a,
        "vehicle_b_progress_info": vehicle_b,
        "anchor_source": "label",
        "base_ratio_a": base_ratio_a,
        "true_ratio_a": float(item["true_ratio_a"]),
    }
    row.update(evidence_to_flat_dict(evidence))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adjustment training CSV using label anchors, detector, and tracking evidence.")
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--class_maps_json", type=Path, default=DEFAULT_CLASS_MAPS)
    parser.add_argument("--base_ratio_csv", type=Path, default=DEFAULT_BASE_RATIO_CSV)
    parser.add_argument("--detector_weights", type=Path, default=DEFAULT_DETECTOR_WEIGHTS)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_missed", type=int, default=5)
    parser.add_argument("--min_track_length", type=int, default=5)
    parser.add_argument("--frame_stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_json(args.input_json)
    class_maps = load_json(args.class_maps_json)
    base_lookup = BaseRatioLookup(args.base_ratio_csv)
    detector = FasterRCNNDetector(args.detector_weights, device=args.device, score_threshold=args.score_threshold)
    tracker = IoUTracker(args.iou_threshold, args.max_missed, args.min_track_length)

    rows = []
    for index, item in enumerate(items, 1):
        try:
            print(f"[{index}/{len(items)}] {item['video_path']}")
            row = process_item(
                item,
                detector,
                tracker,
                base_lookup,
                class_maps,
                args.frame_stride,
            )
            if row is not None:
                rows.append(row)
                print(f"  base={row['base_ratio_a']:.0f}, true={row['true_ratio_a']:.0f}, entry={row['entry_order']}")
        except Exception as exc:
            print(f"  [skip] {exc}")

    if not rows:
        raise RuntimeError("No valid rows were generated.")
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"saved: {args.output_csv}")
    print(f"rows: {len(df)}")
    print(f"mean adjustment: {(df['true_ratio_a'] - df['base_ratio_a']).mean():.3f}")


if __name__ == "__main__":
    main()
