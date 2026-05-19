from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from inference.base_ratio import BaseRatioLookup
from inference.config import BASE_RATIO_CSV, DEVICE
from inference.pipeline import InferencePipeline


@lru_cache(maxsize=1)
def get_pipeline(device: str) -> InferencePipeline:
    return InferencePipeline.from_defaults(device=device)


def accident_places() -> list[str]:
    return BaseRatioLookup.from_csv(BASE_RATIO_CSV).place_options()


def video_filepath(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, tuple) and value:
        return video_filepath(value[0])
    if isinstance(value, dict):
        return video_filepath(value.get("path") or value.get("name") or value.get("video"))
    name = getattr(value, "name", None)
    return str(name) if name else None


def analyze(video_value: Any, accident_place: str, device: str) -> tuple[str | None, str]:
    video_path = video_filepath(video_value)
    if not video_path:
        return None, "영상 파일을 먼저 업로드해 주세요."
    try:
        result = get_pipeline(device).run(video_path, accident_place)
        return result["annotated_video"], result["report"]
    except Exception as exc:
        return None, f"분석 실패: `{exc}`"


def build_ui(device: str) -> gr.Blocks:
    places = accident_places()
    with gr.Blocks(title="교통사고 과실비율 분석") as demo:
        gr.Markdown("# 교통사고 과실비율 분석")
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="사고 영상 업로드", file_types=["video"], type="filepath")
                place_input = gr.Dropdown(places, value=places[0], label="사고 장소")
                run_btn = gr.Button("분석 시작", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="객체 탐지 결과", height=280, format="mp4")
                report_output = gr.Markdown("분석 결과가 여기에 표시됩니다.")
        run_btn.click(lambda video, place: analyze(video, place, device), [video_input, place_input], [video_output, report_output])
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui(args.device).launch(server_name="0.0.0.0", server_port=7860, share=args.share, inbrowser=False, max_file_size="1gb")
