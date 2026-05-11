import json
import random
from pathlib import Path
from typing import Dict, List, Optional


# =========================
# Config
# =========================

BASE_DIR = Path("/home/junhyung/Documents/vscode/car_accident/2026-1-semester-CV-project/classification/video_data")

RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

VIDEO_DIRS = [RAW_DIR / "VS_차대차_영상_직선도로"]

LABEL_DIRS = [RAW_DIR / "VL_차대차_영상_직선도로"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


# =========================
# Utility functions
# =========================

def load_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_all_videos(video_dirs: List[Path]) -> List[Path]:
    video_paths = []

    for video_dir in video_dirs:
        if not video_dir.exists():
            print(f"[Warning] video dir not found: {video_dir}")
            continue

        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(video_dir.rglob(f"*{ext}"))

    return sorted(video_paths)


def find_all_jsons(label_dirs: List[Path]) -> Dict[str, Path]:
    """
    json 파일들을 stem 기준으로 index화한다.
    예:
      bb_1_140725_two-wheeled-vehicle_226_20661.json
      -> key: bb_1_140725_two-wheeled-vehicle_226_20661
    """
    json_index = {}

    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"[Warning] label dir not found: {label_dir}")
            continue

        for json_path in label_dir.rglob("*.json"):
            key = json_path.stem

            if key in json_index:
                print(f"[Warning] duplicated json stem: {key}")
                print(f"  existing: {json_index[key]}")
                print(f"  new     : {json_path}")

            json_index[key] = json_path

    return json_index


def match_video_and_json(video_paths: List[Path], json_index: Dict[str, Path]) -> List[Dict]:
    """
    video 파일명과 json 파일명을 stem 기준으로 매칭한다.
    """
    samples = []
    missing_jsons = []

    for video_path in video_paths:
        key = video_path.stem
        json_path = json_index.get(key)

        if json_path is None:
            missing_jsons.append(video_path)
            continue

        samples.append({
            "video_path": str(video_path),
            "json_path": str(json_path),
        })

    if missing_jsons:
        print(f"[Warning] json not found for {len(missing_jsons)} videos.")
        for p in missing_jsons[:10]:
            print(f"  missing json for: {p}")
        if len(missing_jsons) > 10:
            print("  ...")

    return samples


def extract_labels(json_path: Path) -> Optional[Dict]:
    data = load_json(json_path)

    if "video" not in data:
        raise KeyError(f"'video' key not found in json: {json_path}")

    video_info = data["video"]

    required_keys = [
        "accident_place_feature",
        "vehicle_a_progress_info",
        "vehicle_b_progress_info",
        "traffic_accident_type"
    ]

    for key in required_keys:
        if key not in video_info:
            print(f"[Skip] '{key}' key not found in json: {json_path}")
            return None

    labels = {
        "accident_place_feature": video_info["accident_place_feature"],
        "vehicle_a_progress_info": video_info["vehicle_a_progress_info"],
        "vehicle_b_progress_info": video_info["vehicle_b_progress_info"],
        "traffic_accident_type": video_info["traffic_accident_type"],
    }

    return labels


def build_label_json(samples: List[Dict]) -> List[Dict]:
    labeled_samples = []

    for sample in samples:
        video_path = Path(sample["video_path"])
        json_path = Path(sample["json_path"])

        labels = extract_labels(json_path)

        if labels is None:
            continue

        labeled_sample = {
            "video_path": str(video_path),
            "json_path": str(json_path),
            **labels,
        }

        labeled_samples.append(labeled_sample)

    return labeled_samples


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Dict]]:
    random.seed(seed)

    samples = samples.copy()
    random.shuffle(samples)

    total_size = len(samples)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Collecting videos...")
    video_paths = find_all_videos(VIDEO_DIRS)
    print(f"Found videos: {len(video_paths)}")

    print("[2] Collecting json labels...")
    json_index = find_all_jsons(LABEL_DIRS)
    print(f"Found jsons: {len(json_index)}")

    print("[3] Matching video and json...")
    samples = match_video_and_json(video_paths, json_index)
    print(f"Matched samples: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("No matched samples found. Check video/json filenames and directory paths.")

    print("[4] Filtering valid samples...")
    valid_samples = []

    for sample in samples:
        json_path = Path(sample["json_path"])
        labels = extract_labels(json_path)

        if labels is None:
            continue

        valid_samples.append(sample)

    print(f"Valid samples: {len(valid_samples)}")
    print(f"Skipped samples: {len(samples) - len(valid_samples)}")

    print("[5] Splitting train / val / test...")
    split_data = split_samples(
        samples=valid_samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    split_path = PROCESSED_DIR / "split.json"
    save_json(split_data, split_path)

    print(f"Train samples: {len(split_data['train'])}")
    print(f"Val samples  : {len(split_data['val'])}")
    print(f"Test samples : {len(split_data['test'])}")
    print(f"Saved split file: {split_path}")

    print("[6] Building train.json...")
    train_data = build_label_json(split_data["train"])
    train_path = PROCESSED_DIR / "train.json"
    save_json(train_data, train_path)
    print(f"Saved train file: {train_path}")

    print("[7] Building val.json...")
    val_data = build_label_json(split_data["val"])
    val_path = PROCESSED_DIR / "val.json"
    save_json(val_data, val_path)
    print(f"Saved val file: {val_path}")

    print("[8] Building test.json...")
    test_data = build_label_json(split_data["test"])
    test_path = PROCESSED_DIR / "test.json"
    save_json(test_data, test_path)
    print(f"Saved test file: {test_path}")

    print("[Done]")


if __name__ == "__main__":
    main()