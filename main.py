import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2

import object_tracking
import video_classification


PROJECT_ROOT = Path(__file__).resolve().parent
FRAME_ROOT = PROJECT_ROOT / "tracking" / "frame_images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DETECTOR_WEIGHTS = PROJECT_ROOT / "detect" / "detection_outputs" / "checkpoints" / "faster_rcnn_baseline" / "best.pth"
DEFAULT_CLASSIFIER_WEIGHTS = PROJECT_ROOT / "classification" / "video_classification_outputs" / "best.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/junhyung/Documents/vscode/car_accident/2026-1-semester-CV-project/bb_1_000129_vehicle_228_29089.mp4")
    parser.add_argument("--detector-weights", default=str(DEFAULT_DETECTOR_WEIGHTS))
    parser.add_argument("--classifier-weights", default=str(DEFAULT_CLASSIFIER_WEIGHTS))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def optional_path(value):
    if value is None:
        return None

    value = str(value).strip()
    return Path(value) if value else None


def write_frame(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return False

    encoded.tofile(str(path))
    return True


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


def run(args):
    video_path = optional_path(args.video)
    video_stem = video_path.stem if video_path is not None else "no_video"
    run_name = f"{video_stem}_{args.fps}fps"
    frame_dir = FRAME_ROOT / run_name
    output_dir = optional_path(args.output_dir) or OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(video_path, frame_dir, args.fps)

    tracking_result = object_tracking.run(
        SimpleNamespace(
            video=str(video_path) if video_path is not None else "",
            weights=args.detector_weights,
            fps=args.fps,
            device=args.device,
            score_threshold=0.5,
            iou_threshold=0.3,
            max_missed=5,
            min_track_length=5,
            frame_dir=str(frame_dir),
            output_dir=str(output_dir / "tracking"),
            name=run_name,
        )
    )

    classification_result = video_classification.run(
        SimpleNamespace(
            video=str(video_path) if video_path is not None else "",
            weights=args.classifier_weights,
            device=args.device,
            frame_size=224,
            expected_frames=150,
            output_dir=str(output_dir / "classification"),
            name=run_name,
        )
    )

    tracking_output_path = output_dir / "tracking_result.json"
    classification_output_path = output_dir / "classification_result.json"

    with open(tracking_output_path, "w", encoding="utf-8") as f:
        json.dump(tracking_result, f, ensure_ascii=False, indent=2)

    with open(classification_output_path, "w", encoding="utf-8") as f:
        json.dump(classification_result, f, ensure_ascii=False, indent=2)

    print(f"Tracking result: {tracking_output_path}")
    print(f"Classification result: {classification_output_path}")


if __name__ == "__main__":
    run(parse_args())
