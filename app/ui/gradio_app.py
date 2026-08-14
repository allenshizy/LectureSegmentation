from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from app.config import AppConfig
from app.pipeline.pipeline import Chapter, SegmentationPipeline

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_app(config: AppConfig | None = None) -> gr.Blocks:
    pipeline = SegmentationPipeline(config)

    def run_pipeline(file_path: str, describe_chapters: bool):
        if not file_path:
            raise gr.Error("Please provide a local audio/video file path.")
        source = Path(file_path).expanduser()
        if not source.is_file():
            raise gr.Error(f"File not found: {source}")
        logger.info(
            "Pipeline request received: file=%s, generate_descriptions=%s",
            source,
            describe_chapters,
        )
        try:
            result = pipeline.run(source, describe_chapters=describe_chapters)
        except Exception as exc:
            logger.exception("Pipeline failed for %s", source)
            raise gr.Error(f"Pipeline failed: {exc}") from exc
        logger.info("Pipeline request finished: produced %d chapters", len(result.chapters))
        overview = result.course_summary or "Course overview was not generated. Check the terminal log for Qwen errors."
        return result.chapters, overview

    with gr.Blocks(title="Lecture Segmentation Pipeline") as demo:
        gr.Markdown("# Lecture Segmentation Pipeline\nWhisper (ASR) -> STB (boundary detection) -> Qwen via Ollama (title/keywords/summary)")
        with gr.Row():
            file_path = gr.Textbox(label="Local audio/video file path", placeholder="/path/to/lecture.mp4")
            describe_chapters = gr.Checkbox(label="Generate title/keywords/summary with Qwen", value=True)
        run_button = gr.Button("Run pipeline", variant="primary")
        course_overview = gr.Textbox(
            label="Course overview",
            lines=5,
            max_lines=8,
            interactive=False,
        )
        gr.Markdown("## Detected chapters\nClick a chapter to expand its keywords, summary, and full transcript.")
        chapters_state = gr.State([])

        @gr.render(inputs=chapters_state)
        def render_chapters(chapters: list[Chapter]):
            if not chapters:
                gr.Markdown("_No chapters yet. Run the pipeline above._")
                return
            for index, chapter in enumerate(chapters, start=1):
                title = chapter.title or "Metadata unavailable"
                label = f"{index}. [{_format_time(chapter.start)} - {_format_time(chapter.end)}] {title}"
                with gr.Accordion(label, open=False):
                    gr.Markdown(f"**Keywords:** {', '.join(chapter.keywords or []) or '—'}")
                    gr.Markdown(
                        f"**Summary:** {chapter.summary or chapter.description_error or 'Qwen did not return a valid description'}"
                    )
                    gr.Textbox(
                        value=chapter.text,
                        label="Full transcript",
                        lines=6,
                        max_lines=40,
                        interactive=False,
                        show_copy_button=True,
                    )

        run_button.click(
            fn=run_pipeline,
            inputs=[file_path, describe_chapters],
            outputs=[chapters_state, course_overview],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
