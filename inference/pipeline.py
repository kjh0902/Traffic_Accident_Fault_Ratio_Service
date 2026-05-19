from __future__ import annotations

import tempfile
from pathlib import Path

import cv2

from .adjustment import AdjustmentModel
from .base_ratio import BaseRatioLookup
from .class_maps import decode
from .config import ADJUSTMENT_MODEL, BASE_RATIO_CSV, CLASSIFIER_WEIGHTS, DETECTOR_WEIGHTS, DEVICE
from .detector import FasterRCNNDetector
from .report import generate_report
from .schemas import SceneAnchor
from .tracker import IoUTracker, assign_actors_by_size, tracks_from_objects
from .trajectory import build_evidence
from .video_classifier import VideoClassifier


class InferencePipeline:
    def __init__(
        self,
        classifier: VideoClassifier,
        detector: FasterRCNNDetector,
        base_lookup: BaseRatioLookup,
        adjustment: AdjustmentModel,
        tracker: IoUTracker | None = None,
    ):
        self.classifier = classifier
        self.detector = detector
        self.base_lookup = base_lookup
        self.adjustment = adjustment
        self.tracker = tracker or IoUTracker()

    @classmethod
    def from_defaults(cls, device: str = DEVICE) -> "InferencePipeline":
        return cls(
            VideoClassifier(CLASSIFIER_WEIGHTS, device=device),
            FasterRCNNDetector(DETECTOR_WEIGHTS, device=device),
            BaseRatioLookup.from_csv(BASE_RATIO_CSV),
            AdjustmentModel.load(ADJUSTMENT_MODEL) if ADJUSTMENT_MODEL.exists() else AdjustmentModel.zero(),
        )

    def run(self, video_path: str | Path, accident_place: str) -> dict:
        video_path = Path(video_path)
        raw = self.classifier.predict(video_path)

        def pred(key: str) -> str:
            value = str(raw[key]["class_name"])
            return decode(key, value)

        anchor = SceneAnchor(
            accident_place=accident_place,
            accident_place_feature=pred("accident_place_feature"),
            vehicle_a_progress_info=pred("vehicle_a_progress_info"),
            vehicle_b_progress_info=pred("vehicle_b_progress_info"),
            confidence=min(float(x["score"]) for x in raw.values()),
        )
        base = self.base_lookup.get(anchor)
        detections, fps, width, height = self.detector.detect_video(video_path)
        objects = assign_actors_by_size(self.tracker.track(detections))
        evidence = build_evidence(tracks_from_objects(objects), fps, width, height)
        adjusted = self.adjustment.apply(base.ratio_a, evidence)
        annotated = draw_detections(video_path, detections, fps, width, height)
        return {
            "anchor": anchor,
            "base": base,
            "evidence": evidence,
            "adjusted": adjusted,
            "annotated_video": annotated,
            "report": generate_report(anchor, base, adjusted, evidence),
        }


def draw_detections(video_path: Path, detections: list[list], fps: float, width: int, height: int) -> str:
    cap = cv2.VideoCapture(str(video_path))
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out.name
    out.close()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for det in detections[frame_index] if frame_index < len(detections) else []:
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 220, 80), 2)
            cv2.putText(frame, f"{det.label} {det.score:.2f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 220, 80), 2)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    return out_path

