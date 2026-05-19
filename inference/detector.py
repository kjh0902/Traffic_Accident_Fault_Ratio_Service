from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .schemas import Detection


class FasterRCNNDetector:
    def __init__(self, weights_path: Path, device: str = "cuda:0", score_threshold: float = 0.5):
        ckpt = torch.load(weights_path, map_location=device)
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.classes = tuple(ckpt["classes"])
        self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            int(ckpt.get("num_classes_with_background", len(self.classes) + 1)),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    def label_name(self, label_idx: int) -> str:
        return str(self.classes[label_idx - 1]) if 0 < label_idx <= len(self.classes) else str(label_idx)

    @torch.inference_mode()
    def detect_frame(self, frame, frame_index: int) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.model([TF.to_tensor(rgb).to(self.device)])[0]
        detections = []
        for box, score, label_idx in zip(output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()):
            if float(score) >= self.score_threshold:
                detections.append(Detection(frame_index, tuple(map(float, box.tolist())), float(score), self.label_name(int(label_idx))))
        return detections

    def detect_video(self, video_path: str | Path, frame_stride: int = 1) -> tuple[list[list[Detection]], float, int, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detections, frame_index = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections.append(self.detect_frame(frame, frame_index) if frame_index % frame_stride == 0 else [])
            frame_index += 1
        cap.release()
        return detections, fps, width, height

