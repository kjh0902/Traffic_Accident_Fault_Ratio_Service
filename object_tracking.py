import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
FRAME_ROOT = PROJECT_ROOT / "tracking" / "frame_images"
OUTPUT_ROOT = PROJECT_ROOT / "tracking" / "tracking_outputs"
DEFAULT_WEIGHTS = PROJECT_ROOT / "detect" / "detection_outputs" / "checkpoints" / "faster_rcnn_baseline" / "best.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", nargs="?", const="", default="")
    parser.add_argument("--weights", nargs="?", const="", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", nargs="?", const="", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--min-track-length", type=int, default=5)
    parser.add_argument("--frame-dir", nargs="?", const="", default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def optional_path(value):
    if value is None:
        return None

    value = str(value).strip()
    return Path(value) if value else None


def write_result(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tracking_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(output_path)
    return result


def write_frame(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return False

    encoded.tofile(str(path))
    return True


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    num_classes = checkpoint["num_classes_with_background"]

    model = get_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    label_names = {idx + 1: name for idx, name in enumerate(checkpoint["classes"])}
    return model, label_names


def extract_frames(video_path, frame_dir, target_fps):
    frame_dir.mkdir(parents=True, exist_ok=True)
    for image_path in frame_dir.glob("*.jpg"):
        image_path.unlink()

    if video_path is None or not video_path.is_file():
        return []

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        cap.release()
        return []

    interval = 1
    if source_fps > 0 and target_fps > 0:
        interval = max(1, round(source_fps / target_fps))

    frame_paths = []
    frame_idx = 0
    save_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % interval == 0:
            path = frame_dir / f"frame_{save_idx:06d}.jpg"
            if write_frame(path, frame):
                frame_paths.append(path)
                save_idx += 1

        frame_idx += 1

    cap.release()
    return frame_paths


@torch.no_grad()
def detect(model, image_path, device, score_threshold):
    image = Image.open(image_path).convert("RGB")
    output = model([F.to_tensor(image).to(device)])[0]

    detections = []
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if score.item() < score_threshold:
            continue

        detections.append(
            {
                "bbox": [round(v, 2) for v in box.cpu().tolist()],
                "score": round(score.item(), 4),
                "label": int(label.item()),
            }
        )

    return detections


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def update_tracks(tracks, finished_tracks, detections, frame_idx, next_id, iou_threshold, max_missed):
    used_detection_ids = set()

    for track in tracks:
        best_det_id = None
        best_iou = 0

        for det_id, det in enumerate(detections):
            if det_id in used_detection_ids or det["label"] != track["label"]:
                continue

            score = iou(track["bbox"], det["bbox"])
            if score > best_iou:
                best_iou = score
                best_det_id = det_id

        if best_det_id is not None and best_iou >= iou_threshold:
            det = detections[best_det_id]
            track["bbox"] = det["bbox"]
            track["last_frame"] = frame_idx
            track["missed"] = 0
            track["history"].append(
                {
                    "frame": frame_idx,
                    "bbox": det["bbox"],
                    "score": det["score"],
                }
            )
            used_detection_ids.add(best_det_id)
        else:
            track["missed"] += 1

    alive_tracks = []
    for track in tracks:
        if track["missed"] > max_missed:
            finished_tracks.append(track)
        else:
            alive_tracks.append(track)
    tracks[:] = alive_tracks

    for det_id, det in enumerate(detections):
        if det_id in used_detection_ids:
            continue

        tracks.append(
            {
                "track_id": next_id,
                "label": det["label"],
                "bbox": det["bbox"],
                "start_frame": frame_idx,
                "last_frame": frame_idx,
                "missed": 0,
                "history": [
                    {
                        "frame": frame_idx,
                        "bbox": det["bbox"],
                        "score": det["score"],
                    }
                ],
            }
        )
        next_id += 1

    return next_id


def run(args):
    video_path = optional_path(args.video)
    weights_path = optional_path(args.weights)
    video_stem = video_path.stem if video_path is not None else "no_video"
    run_name = args.name or f"{video_stem}_{args.fps}fps"
    frame_dir_arg = optional_path(getattr(args, "frame_dir", None))
    frame_dir = frame_dir_arg or FRAME_ROOT / run_name
    output_dir = optional_path(getattr(args, "output_dir", None)) or OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    frame_paths = sorted(frame_dir.glob("*.jpg")) if frame_dir_arg is not None else extract_frames(video_path, frame_dir, args.fps)
    skip_reasons = []
    if frame_dir_arg is None and (video_path is None or not video_path.is_file()):
        skip_reasons.append("video path is empty or the video file does not exist")
    if not frame_paths:
        skip_reasons.append("no frames were found or extracted")
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
            "fps": args.fps,
            "frame_dir": str(frame_dir),
            "frame_count": len(frame_paths),
            "label_names": {},
            "tracks": [],
        }
        return write_result(output_dir, result)

    try:
        model, label_names = load_detector(weights_path, device)
    except Exception as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "video": str(video_path) if video_path is not None else "",
            "weights": str(weights_path),
            "fps": args.fps,
            "frame_dir": str(frame_dir),
            "frame_count": len(frame_paths),
            "label_names": {},
            "tracks": [],
        }
        return write_result(output_dir, result)

    tracks = []
    finished_tracks = []
    next_id = 1

    for frame_idx, frame_path in enumerate(tqdm(frame_paths)):
        detections = detect(model, frame_path, device, args.score_threshold)
        next_id = update_tracks(
            tracks,
            finished_tracks,
            detections,
            frame_idx,
            next_id,
            args.iou_threshold,
            args.max_missed,
        )

    result = {
        "status": "ok",
        "video": str(video_path),
        "weights": str(weights_path),
        "fps": args.fps,
        "frame_dir": str(frame_dir),
        "frame_count": len(frame_paths),
        "label_names": label_names,
        "tracks": [
            {
                "track_id": track["track_id"],
                "label": track["label"],
                "label_name": label_names.get(track["label"]),
                "start_frame": track["start_frame"],
                "last_frame": track["last_frame"],
                "history": track["history"],
            }
            for track in finished_tracks + tracks
            if len(track["history"]) >= args.min_track_length
        ],
    }

    return write_result(output_dir, result)


if __name__ == "__main__":
    run(parse_args())
