from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.video import r2plus1d_18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class R2Plus1DMultiHeadClassifier(nn.Module):
    def __init__(self, num_classes: dict[str, int]):
        super().__init__()
        self.backbone = r2plus1d_18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({key: nn.Linear(in_features, n) for key, n in num_classes.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x.permute(0, 2, 1, 3, 4).contiguous())
        return {key: head(feat) for key, head in self.heads.items()}


def read_video_tensor(video_path: Path, frame_size: int, expected_frames: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frames.append(TF.normalize(TF.to_tensor(frame), IMAGENET_MEAN, IMAGENET_STD))
    cap.release()

    if not frames:
        frames = [torch.zeros(3, frame_size, frame_size)]
    if len(frames) > expected_frames:
        indices = torch.linspace(0, len(frames) - 1, expected_frames).long().tolist()
        frames = [frames[i] for i in indices]
    while len(frames) < expected_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames).unsqueeze(0)


class VideoClassifier:
    def __init__(self, weights_path: Path, device: str = "cuda:0", frame_size: int = 224, expected_frames: int = 150):
        ckpt = torch.load(weights_path, map_location=device)
        self.device = torch.device(device)
        self.frame_size = frame_size
        self.expected_frames = expected_frames
        self.label_mappings: dict[str, dict[str, Any]] = ckpt["label_mappings"]
        num_classes = {
            key: int(value.get("num_classes", len(value["idx_to_label"])))
            for key, value in self.label_mappings.items()
        }
        self.model = R2Plus1DMultiHeadClassifier(num_classes).to(self.device)
        self.model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict")))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, video_path: str | Path) -> dict[str, dict[str, Any]]:
        video = read_video_tensor(Path(video_path), self.frame_size, self.expected_frames).to(self.device)
        result = {}
        for key, logits in self.model(video).items():
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
            labels = self.label_mappings[key]["idx_to_label"]
            result[key] = {"class_name": labels.get(idx, labels.get(str(idx), idx)), "score": float(probs[idx].item())}
        return result

