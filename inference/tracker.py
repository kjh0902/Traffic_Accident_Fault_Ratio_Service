from __future__ import annotations

from collections import defaultdict

from .schemas import Box, Detection, Track, TrackedObject


def iou(a: Box, b: Box) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, min_track_length: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_track_length = min_track_length

    def track(self, detections_by_frame: list[list[Detection]]) -> list[TrackedObject]:
        active, finished, next_id = [], [], 1
        for frame_dets in detections_by_frame:
            used = set()
            for track in active:
                candidates = [
                    (i, iou(track["bbox"], det.bbox_xyxy))
                    for i, det in enumerate(frame_dets)
                    if i not in used and det.label == track["label"]
                ]
                best = max(candidates, key=lambda x: x[1], default=(None, 0.0))
                if best[0] is not None and best[1] >= self.iou_threshold:
                    det = frame_dets[best[0]]
                    track.update(bbox=det.bbox_xyxy, missed=0)
                    track["history"].append(det)
                    used.add(best[0])
                else:
                    track["missed"] += 1
            finished += [t for t in active if t["missed"] > self.max_missed]
            active = [t for t in active if t["missed"] <= self.max_missed]
            for i, det in enumerate(frame_dets):
                if i not in used:
                    active.append({"track_id": next_id, "label": det.label, "bbox": det.bbox_xyxy, "missed": 0, "history": [det]})
                    next_id += 1

        objects = []
        for track in finished + active:
            if len(track["history"]) >= self.min_track_length:
                objects += [
                    TrackedObject(det.frame_index, track["track_id"], det.bbox_xyxy, det.score, det.label)
                    for det in track["history"]
                ]
        return objects


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def assign_actors_by_size(objects: list[TrackedObject]) -> list[TrackedObject]:
    first_seen = {}
    for obj in sorted(objects, key=lambda x: x.frame_index):
        first_seen.setdefault(obj.track_id, obj)
    ranked = sorted(first_seen.values(), key=lambda x: box_area(x.bbox_xyxy), reverse=True)
    actor_map = {obj.track_id: actor for obj, actor in zip(ranked, ("A", "B"))}
    return [TrackedObject(o.frame_index, o.track_id, o.bbox_xyxy, o.score, o.label, actor_map.get(o.track_id)) for o in objects]


def tracks_from_objects(objects: list[TrackedObject]) -> list[Track]:
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj.track_id].append(obj)
    return [
        Track(track_id, rows[0].label, next((x.actor for x in rows if x.actor), None), sorted(rows, key=lambda x: x.frame_index))
        for track_id, rows in grouped.items()
    ]

