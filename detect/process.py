# detect/process.py

import json
import random
from pathlib import Path

from PIL import Image


# =========================================================
# Config
# =========================================================

configs = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}


# =========================================================
# 경로 설정
# =========================================================

PROJECT_ROOT = Path("/home/junhyung/Documents/vscode/car_accident/2026-1-semester-CV-project")

DETECT_ROOT = PROJECT_ROOT / "detect"

RAW_ROOT = DETECT_ROOT / "img_data" / "raw"
PROCESSED_ROOT = DETECT_ROOT / "img_data" / "processed"
OUTPUT_ROOT = DETECT_ROOT / "detection_outputs"

IMAGE_DIR = RAW_ROOT / "VS_차대차_이미지_T자형교차로"
LABEL_DIR = RAW_ROOT / "VL_차대차_이미지_T자형교차로"

TRAIN_SAMPLES_PATH = PROCESSED_ROOT / "train_samples.json"
VAL_SAMPLES_PATH = PROCESSED_ROOT / "val_samples.json"
TEST_SAMPLES_PATH = PROCESSED_ROOT / "test_samples.json"

SPLIT_JSON_PATH = PROCESSED_ROOT / "split.json"

TRAIN_COCO_PATH = PROCESSED_ROOT / "train_coco.json"
VAL_COCO_PATH = PROCESSED_ROOT / "val_coco.json"
TEST_COCO_PATH = PROCESSED_ROOT / "test_coco.json"

CLASS_MAP_PATH = PROCESSED_ROOT / "class_map.json"
IDX_TO_CLASS_PATH = PROCESSED_ROOT / "idx_to_class.json"


# =========================================================
# 기본 유틸
# =========================================================

def ensure_dirs():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def relative_to_project(path):
    path = Path(path)
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def print_paths():
    print("[경로 확인]")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("IMAGE_DIR:", IMAGE_DIR)
    print("LABEL_DIR:", LABEL_DIR)
    print("PROCESSED_ROOT:", PROCESSED_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("IMAGE_DIR exists?", IMAGE_DIR.exists())
    print("LABEL_DIR exists?", LABEL_DIR.exists())
    print("TRAIN_COCO_PATH:", TRAIN_COCO_PATH)
    print("VAL_COCO_PATH:", VAL_COCO_PATH)
    print("TEST_COCO_PATH:", TEST_COCO_PATH)


# =========================================================
# annotation parsing
# =========================================================

def parse_annotation(ann):
    """
    annotation json에서 object 정보를 파싱한다.

    현재 가정:
      ann["objects"] 안에 object들이 있고,
      각 object는 bbox, category를 가진다.

    원본 bbox:
      [x, y, w, h]

    반환 bbox:
      [x1, y1, x2, y2]
    """

    objs = []

    for obj in ann.get("objects", []):
        bbox = obj.get("bbox", None)
        category = obj.get("category", None)

        if bbox is None or len(bbox) != 4:
            continue

        if category is None:
            continue

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            continue

        objs.append(
            {
                "bbox": [x, y, x + w, y + h],
                "label": category,
            }
        )

    return objs


# =========================================================
# 이미지-라벨 매칭
# =========================================================

def find_json_for_image(img_path, image_dir, label_dir):
    """
    이미지에 대응하는 json 경로를 찾는다.

    1. label_dir / 이미지파일명.json
    2. label_dir 내부에서 이미지 상대경로 유지 후 .json
    3. 하위 폴더 전체에서 stem 기준 검색
    """

    img_path = Path(img_path)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # 1) label_dir / stem.json
    candidate1 = label_dir / f"{img_path.stem}.json"
    if candidate1.exists():
        return candidate1

    # 2) 상대경로 유지
    try:
        rel = img_path.relative_to(image_dir).with_suffix(".json")
        candidate2 = label_dir / rel
        if candidate2.exists():
            return candidate2
    except Exception:
        pass

    # 3) label_dir 전체에서 stem.json 검색
    candidates = list(label_dir.rglob(f"{img_path.stem}.json"))
    if len(candidates) > 0:
        return candidates[0]

    return None


# =========================================================
# sample 생성
# =========================================================

def build_samples(image_dir, label_dir, image_exts=(".png", ".jpg", ".jpeg", ".bmp")):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(
            f"라벨 폴더가 없습니다: {label_dir}\n"
            f"현재 LABEL_DIR은 직접 지정된 경로를 사용합니다.\n"
            f"실제 라벨 폴더명이 다르면 process.py의 LABEL_DIR을 수정하세요."
        )

    image_files = sorted(
        [
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]
    )

    samples = []
    missing_json = []
    empty_annotations = []
    parse_failed = []

    for img_path in image_files:
        json_path = find_json_for_image(img_path, image_dir, label_dir)

        if json_path is None or not json_path.exists():
            missing_json.append(str(img_path))
            continue

        try:
            ann = load_json(json_path)
            objects = parse_annotation(ann)
        except Exception as e:
            parse_failed.append(
                {
                    "image_path": str(img_path),
                    "json_path": str(json_path),
                    "error": str(e),
                }
            )
            continue

        if len(objects) == 0:
            empty_annotations.append(str(img_path))
            continue

        samples.append(
            {
                "image_path": str(img_path),
                "json_path": str(json_path),
            }
        )

    info = {
        "num_images_found": len(image_files),
        "num_valid_samples": len(samples),
        "num_missing_json": len(missing_json),
        "num_empty_annotations": len(empty_annotations),
        "num_parse_failed": len(parse_failed),
        "missing_json_examples": missing_json[:20],
        "empty_annotation_examples": empty_annotations[:20],
        "parse_failed_examples": parse_failed[:20],
    }

    return samples, info


# =========================================================
# class map 생성
# =========================================================

def build_class_map(samples):
    label_names = set()

    for sample in samples:
        ann = load_json(sample["json_path"])
        objects = parse_annotation(ann)

        for obj in objects:
            label_names.add(obj["label"])

    label_names = sorted(label_names)

    # Faster R-CNN에서 background가 0이므로 class id는 1부터 시작
    class_map = {
        name: idx + 1
        for idx, name in enumerate(label_names)
    }

    idx_to_class = {
        idx: name
        for name, idx in class_map.items()
    }

    return class_map, idx_to_class


# =========================================================
# train / val / test split
# =========================================================

def split_samples(
    samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    if len(samples) == 0:
        raise ValueError("samples가 비어 있습니다.")

    if len(samples) < 3:
        raise ValueError(
            "train/val/test로 나누려면 최소 3개 이상의 sample이 필요합니다."
        )

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio는 1.0이어야 합니다. 현재: {ratio_sum}"
        )

    random.seed(seed)

    samples_copy = samples.copy()
    random.shuffle(samples_copy)

    n_total = len(samples_copy)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # val/test가 0개가 되는 것 방지
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    # test도 최소 1개 남기기
    if n_train + n_val >= n_total:
        n_train = n_total - 2
        n_val = 1

    train_samples = samples_copy[:n_train]
    val_samples = samples_copy[n_train:n_train + n_val]
    test_samples = samples_copy[n_train + n_val:]

    return train_samples, val_samples, test_samples


# =========================================================
# COCO 변환
# =========================================================

def clip_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box

    x1 = max(0, min(float(x1), width - 1))
    y1 = max(0, min(float(y1), height - 1))
    x2 = max(0, min(float(x2), width - 1))
    y2 = max(0, min(float(y2), height - 1))

    return [x1, y1, x2, y2]


def convert_to_coco(samples, class_map, output_json):
    images = []
    annotations = []
    categories = []

    ann_id = 1

    for name, cid in class_map.items():
        categories.append(
            {
                "id": cid,
                "name": name,
            }
        )

    for img_id, sample in enumerate(samples):
        image_path = Path(sample["image_path"])
        json_path = Path(sample["json_path"])

        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": img_id,
                "file_name": relative_to_project(image_path),
                "width": width,
                "height": height,
            }
        )

        ann = load_json(json_path)
        objects = parse_annotation(ann)

        for obj in objects:
            label_name = obj["label"]

            if label_name not in class_map:
                continue

            x1, y1, x2, y2 = clip_box_xyxy(obj["bbox"], width, height)

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_map[label_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    save_json(coco, output_json)

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
    }


# =========================================================
# main
# =========================================================

def main():
    ensure_dirs()
    print_paths()

    samples, info = build_samples(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
    )

    print("\n[샘플 빌드 결과]")
    for k, v in info.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)}개")
        else:
            print(f"{k}: {v}")

    if len(samples) == 0:
        raise ValueError(
            "유효한 sample이 0개입니다. IMAGE_DIR, LABEL_DIR, annotation 구조를 확인하세요."
        )

    class_map, idx_to_class = build_class_map(samples)

    print("\n[클래스 정보]")
    print("num_classes:", len(class_map))
    print("class_map:", class_map)

    save_json(class_map, CLASS_MAP_PATH)
    save_json(idx_to_class, IDX_TO_CLASS_PATH)
    save_json(configs, PROCESSED_ROOT / "process_configs.json")

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=configs["train_ratio"],
        val_ratio=configs["val_ratio"],
        test_ratio=configs["test_ratio"],
        seed=configs["seed"],
    )

    save_json(train_samples, TRAIN_SAMPLES_PATH)
    save_json(val_samples, VAL_SAMPLES_PATH)
    save_json(test_samples, TEST_SAMPLES_PATH)

    save_json(
        {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        },
        SPLIT_JSON_PATH,
    )

    print("\n[split 저장 완료]")
    print("train:", len(train_samples))
    print("val:", len(val_samples))
    print("test:", len(test_samples))
    print("saved:", TRAIN_SAMPLES_PATH)
    print("saved:", VAL_SAMPLES_PATH)
    print("saved:", TEST_SAMPLES_PATH)
    print("saved:", SPLIT_JSON_PATH)

    train_info = convert_to_coco(
        samples=train_samples,
        class_map=class_map,
        output_json=TRAIN_COCO_PATH,
    )

    val_info = convert_to_coco(
        samples=val_samples,
        class_map=class_map,
        output_json=VAL_COCO_PATH,
    )

    test_info = convert_to_coco(
        samples=test_samples,
        class_map=class_map,
        output_json=TEST_COCO_PATH,
    )

    print("\n[COCO 변환 완료]")
    print("train coco:", TRAIN_COCO_PATH)
    print("train info:", train_info)
    print("val coco:", VAL_COCO_PATH)
    print("val info:", val_info)
    print("test coco:", TEST_COCO_PATH)
    print("test info:", test_info)

    print("\n[class map 저장 완료]")
    print("class_map:", CLASS_MAP_PATH)
    print("idx_to_class:", IDX_TO_CLASS_PATH)

    print("\n[완료]")


if __name__ == "__main__":
    main()