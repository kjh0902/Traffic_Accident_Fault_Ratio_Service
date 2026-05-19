from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUSTMENT_ROOT = PROJECT_ROOT / "table"
RAW_ROOT = ADJUSTMENT_ROOT / "adjustment_data" / "raw"
PROCESSED_ROOT = ADJUSTMENT_ROOT / "adjustment_data" / "processed"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

ACCIDENT_PLACE = {
    0: "직선 도로",
    1: "사거리교차로(신호등 없음)",
    2: "사거리교차로(신호등 있음)",
    3: "T자형 교차로",
    4: "차도와 차도가 아닌 장소",
    5: "주차장(또는 차도가 아닌 장소)",
    6: "회전교차로",
    7: "횡단보도(신호등 없음)",
    8: "횡단보도(신호등 있음)",
    9: "횡단보도 없음",
    10: "횡단보도(신호등 없음) 부근",
    11: "횡단보도(신호등 있음) 부근",
    12: "육교 및 지하도 부근",
    13: "고속도로(자동차 전용도로)포함",
    14: "자전거 도로",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adjustment_input.json from AI Hub video labels.")
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=RAW_ROOT / "VS_차대차_영상_직선도로",
        help="Directory containing accident videos.",
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=RAW_ROOT / "VL_차대차_영상_직선도로",
        help="Directory containing video label JSON files.",
    )
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        default=None,
        help="Optional directory containing A/B annotation JSON files with the same stems.",
    )
    parser.add_argument(
        "--class_maps_py",
        type=Path,
        default=None,
        help="Optional class_maps.py used to decode classifier code predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_ROOT / "adjustment_input.json",
    )
    parser.add_argument(
        "--class_maps_output",
        type=Path,
        default=PROCESSED_ROOT / "class_maps.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_files_by_stem(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.setdefault(path.stem, path)
    return files


def resolve_class_maps_path(path: Path | None) -> Path:
    candidates = [
        path,
        ADJUSTMENT_ROOT / "class_maps.py",
        PROJECT_ROOT / "src" / "accident_liability" / "scene" / "class_maps.py",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "class_maps.py not found. Pass --class_maps_py, or place class_maps.py in adjustment/."
    )


def load_class_maps(path: Path | None) -> dict[str, dict[str, str]]:
    path = resolve_class_maps_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"class_maps.py not found: {path}\n"
            "Pass --class_maps_py or copy src/accident_liability/scene/class_maps.py."
        )
    spec = importlib.util.spec_from_file_location("adjustment_class_maps", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import class maps: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["adjustment_class_maps"] = module
    spec.loader.exec_module(module)
    return {
        "accident_place": {str(k): v for k, v in ACCIDENT_PLACE.items()},
        "accident_place_feature": dict(module.ACCIDENT_PLACE_FEATURE),
        "vehicle_a_progress_info": dict(module.VEHICLE_A_PROGRESS_INFO),
        "vehicle_b_progress_info": dict(module.VEHICLE_B_PROGRESS_INFO),
    }


def get_video_info(label_path: Path) -> dict:
    data = load_json(label_path)
    return data.get("video", data)


def main() -> None:
    args = parse_args()

    video_map = find_files_by_stem(args.video_dir, VIDEO_EXTENSIONS)
    label_map = find_files_by_stem(args.label_dir, (".json",))
    annotation_map = (
        find_files_by_stem(args.annotation_dir, (".json",))
        if args.annotation_dir is not None and args.annotation_dir.exists()
        else {}
    )

    items = []
    skipped = 0
    for stem, label_path in sorted(label_map.items()):
        video_path = video_map.get(stem)
        if video_path is None:
            skipped += 1
            print(f"[skip] video not found: {stem}")
            continue

        info = get_video_info(label_path)
        place_code = info.get("accident_place")
        true_ratio_a = info.get("accident_negligence_rateA")
        if place_code is None or true_ratio_a is None:
            skipped += 1
            print(f"[skip] required label missing: {stem}")
            continue

        accident_place = ACCIDENT_PLACE.get(int(place_code))
        if accident_place is None:
            skipped += 1
            print(f"[skip] unknown accident_place={place_code}: {stem}")
            continue

        item = {
            "video_path": str(video_path.resolve()),
            "label_path": str(label_path.resolve()),
            "accident_place": accident_place,
            "true_ratio_a": int(true_ratio_a),
        }
        if stem in annotation_map:
            item["annotation_path"] = str(annotation_map[stem].resolve())
        items.append(item)

    class_maps = load_class_maps(args.class_maps_py)
    save_json(items, args.output)
    save_json(class_maps, args.class_maps_output)

    print(f"videos: {len(video_map)}")
    print(f"labels: {len(label_map)}")
    print(f"items: {len(items)} -> {args.output}")
    print(f"class maps -> {args.class_maps_output}")
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
