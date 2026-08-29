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

    def run_pipeline(input_source: str, describe_chapters: bool):
        if not input_source:
            raise gr.Error("Please provide either a YouTube URL or local audio/video file path.")
        
        # Check if input is a YouTube URL
        is_youtube = pipeline._is_youtube_url(input_source)
        
        if is_youtube:
            logger.info(
                "Pipeline request received: youtube_url=%s, generate_descriptions=%s",
                input_source,
                describe_chapters,
            )
            try:
                result = pipeline.run(input_source, describe_chapters=describe_chapters)
                
                # Add YouTube info to the output
                youtube_status = ""
                if result.youtube_info:
                    yt_info = result.youtube_info
                    license_status = "✓ CC-BY compatible" if yt_info.license_allowed else "✗ Non-CC-BY license"
                    subtitle_status = f"{yt_info.subtitle_type} subtitles available"
                    
                    youtube_status = (
                        f"\n\n**YouTube Video Info:**\n"
                        f"- Title: {yt_info.title}\n"
                        f"- Duration: {_format_time(yt_info.duration)}\n"
                        f"- License: {yt_info.license_info or 'Not specified'} ({license_status})\n"
                        f"- Subtitles: {subtitle_status if yt_info.has_subtitles else 'None found'}\n"
                    )
                    
                    if result.subtitle_used:
                        youtube_status += "- **ASR Method: YouTube Subtitles (Whisper skipped)**\n"
                    else:
                        youtube_status += "- **ASR Method: Whisper (No subtitles, audio downloaded)**\n"
            except Exception as exc:
                logger.exception("Pipeline failed for YouTube URL %s", input_source)
                raise gr.Error(f"Pipeline failed: {exc}") from exc
        else:
            # Local file
            source = Path(input_source).expanduser()
            if not source.is_file():
                raise gr.Error(f"File not found: {source}")
            logger.info(
                "Pipeline request received: file=%s, generate_descriptions=%s",
                source,
                describe_chapters,
            )
            try:
                result = pipeline.run(source, describe_chapters=describe_chapters)
                youtube_status = ""
            except Exception as exc:
                logger.exception("Pipeline failed for %s", source)
                raise gr.Error(f"Pipeline failed: {exc}") from exc
        
        logger.info("Pipeline request finished: produced %d chapters", len(result.chapters))
        overview = (result.course_summary or "Course overview was not generated. Check the terminal log for Qwen errors.") + youtube_status
        return result.chapters, overview

    with gr.Blocks(title="Lecture Segmentation Pipeline") as demo:
        gr.Markdown("# Lecture Segmentation Pipeline\nWhisper (ASR) → STB (boundary detection) → Qwen via Ollama (title/keywords/summary)\n\n**Now with YouTube support!** Just paste a YouTube URL or local file path.")
        with gr.Row():
            input_source = gr.Textbox(
                label="YouTube URL or local audio/video file path",
                placeholder="https://www.youtube.com/watch?v=... or /path/to/lecture.mp4"
            )
            describe_chapters = gr.Checkbox(label="Generate title/keywords/summary with Qwen", value=True)
        run_button = gr.Button("Run pipeline", variant="primary")
        course_overview = gr.Textbox(
            label="Course overview & Video info",
            lines=8,
            max_lines=15,
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
            inputs=[input_source, describe_chapters],
            outputs=[chapters_state, course_overview],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
