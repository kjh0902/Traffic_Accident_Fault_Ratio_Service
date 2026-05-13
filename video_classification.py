import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.video import r2plus1d_18


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "classification" / "video_classification_outputs" / "predictions"
DEFAULT_WEIGHTS = PROJECT_ROOT / "classification" / "video_classification_outputs" / "best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", nargs="?", const="", default="")
    parser.add_argument("--weights", nargs="?", const="", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", nargs="?", const="", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--expected-frames", type=int, default=150)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def optional_path(value):
    if value is None:
        return None

    value = str(value).strip()
    return Path(value) if value else None


def write_result(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(output_path)
    return result


class R2Plus1DMultiHeadClassifier(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()

        self.backbone = r2plus1d_18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.heads = nn.ModuleDict()
        for key, num_classes in num_classes_dict.items():
            self.heads[key] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.backbone(x)
        return {key: head(feat) for key, head in self.heads.items()}


def read_video(video_path, frame_size, expected_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frame = TF.to_tensor(frame)
        frame = TF.normalize(frame, IMAGENET_MEAN, IMAGENET_STD)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(3, frame_size, frame_size)]

    if len(frames) > expected_frames:
        indices = torch.linspace(0, len(frames) - 1, expected_frames).long().tolist()
        frames = [frames[i] for i in indices]

    while len(frames) < expected_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=0).unsqueeze(0)


def load_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    label_mappings = checkpoint["label_mappings"]

    num_classes_dict = {
        key: value["num_classes"]
        for key, value in label_mappings.items()
    }

    model = R2Plus1DMultiHeadClassifier(num_classes_dict)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, label_mappings


@torch.no_grad()
def predict(model, label_mappings, video_tensor, device):
    video_tensor = video_tensor.to(device)
    outputs = model(video_tensor)

    result = {}
    for key, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        idx_to_label = label_mappings[key]["idx_to_label"]

        pred_label = idx_to_label.get(pred_idx, idx_to_label.get(str(pred_idx)))

        result[key] = {
            "class_name": pred_label,
            "score": round(float(probs[pred_idx].item()), 4),
        }

    return result


def run(args):
    video_path = optional_path(args.video)
    weights_path = optional_path(args.weights)
    video_stem = video_path.stem if video_path is not None else "no_video"
    run_name = args.name or video_stem
    output_dir = optional_path(getattr(args, "output_dir", None)) or OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    skip_reasons = []
    if video_path is None or not video_path.is_file():
        skip_reasons.append("video path is empty or the video file does not exist")
    if weights_path is None:
        skip_reasons.append("weights path is empty")
    elif not weights_path.is_file():
        skip_reasons.append(f"weights file does not exist: {weights_path}")

    if skip_reasons:
        result = {
            "status": "skipped",
            "skip_reasons": skip_reasons,
            "video": str(video_path) if video_path is not None else "",
            "weights": str(weights_path) if weights_path is not None else "",
            "predictions": {},
        }
        return write_result(output_dir, result)

    try:
        model, label_mappings = load_model(weights_path, device)
        video_tensor = read_video(video_path, args.frame_size, args.expected_frames)
        predictions = predict(model, label_mappings, video_tensor, device)
    except Exception as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "video": str(video_path),
            "weights": str(weights_path),
            "predictions": {},
        }
        return write_result(output_dir, result)

    result = {
        "status": "ok",
        "video": str(video_path),
        "weights": str(weights_path),
        "predictions": predictions,
    }

    return write_result(output_dir, result)


if __name__ == "__main__":
    run(parse_args())
