from __future__ import annotations

import gradio as gr

from app.config import AppConfig
from app.pipeline.pipeline import Chapter, SegmentationPipeline


def _format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _chapters_to_rows(chapters: list[Chapter]) -> list[list[str]]:
    return [
        [_format_time(ch.start), _format_time(ch.end), ch.title or "", ch.text]
        for ch in chapters
    ]


def build_app(config: AppConfig | None = None) -> gr.Blocks:
    pipeline = SegmentationPipeline(config)

    def run_pipeline(file_path: str, title_chapters: bool):
        if not file_path:
            raise gr.Error("Please provide a local audio/video file path.")
        chapters = pipeline.run(file_path, title_chapters=title_chapters)
        return _chapters_to_rows(chapters)

    with gr.Blocks(title="Lecture Segmentation Pipeline") as demo:
        gr.Markdown("# Lecture Segmentation Pipeline\nWhisper (ASR) -> STB (boundary detection) -> Qwen via Ollama (titling)")
        with gr.Row():
            file_path = gr.Textbox(label="Local audio/video file path", placeholder="/path/to/lecture.mp4")
            title_chapters = gr.Checkbox(label="Generate chapter titles with Qwen", value=True)
        run_button = gr.Button("Run pipeline", variant="primary")
        output_table = gr.Dataframe(
            headers=["start", "end", "title", "text"],
            label="Detected chapters",
            wrap=True,
        )
        run_button.click(fn=run_pipeline, inputs=[file_path, title_chapters], outputs=output_table)

    return demo


if __name__ == "__main__":
    build_app().launch()
