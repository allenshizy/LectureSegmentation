from kpi.models.base_model import Model
from kpi.utils.gpt_http import complete_4o
from kpi.utils.video import Video, Srt
from typing import Literal
from textwrap import dedent
import click
from kpi.cli import register_model_runner


def _get_mmss(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02}"


def get_srt_text(srts: list[Srt]):
    return "\n\n".join(
        [
            f"{_get_mmss(s.start)}\n{s.text.replace('\n', ' ')}"
            for _, s in enumerate(srts)
        ]
    )


def call_llm(prompt: str) -> str:
    ret = complete_4o(prompt)
    return ret


class PromptTemplate:
    def get_prompt(self, k: int | None, srts: list[Srt]):
        raise NotImplementedError

    def parse_result(self, result: str, srts: list[Srt]):
        raise NotImplementedError


class BPPromptTemplate(PromptTemplate):
    def get_prompt(self, k: int | None, srts: list[Srt]):
        return dedent(
            f"""\
        Below is the transcript of a lecture video. Please split the video into {k if k else ""} fragments with different topics.
        Transcript is in the following format:
        ---
        <MM:SS>
        <Transcript>
        ---

        Output requirement: You should only output the timestamp of the start of each fragment following MM:SS.
        Transcript:
            """
        ) + get_srt_text(srts)

    def parse_result(self, result: str, srts: list[Srt]):
        ret = []
        for line in result.strip().split("\n"):
            mm, ss = map(int, line.split(":"))
            ret.append(mm * 60 + ss)
        return ret


class APPromptTemplate(PromptTemplate):
    def get_prompt(self, k: int | None, srts: list[Srt]):
        return (
            dedent(
                f"""\
        You are a teaching assistant helping a professor to prepare a lecture video.
        The task is to segment the video into {k if k else ""} fragments with different topics.
        Below is the transcript of a lecture video. Please split the video into fragments with different topics.

        Instructions:
        - You should read and understand the transcript first.
        - You should carefully think and analyze the ENTIRE lecture transcript.
        - You should identify the main topics and subtopics in the lecture.
        - You should summarize the main topics and subtopics.
        - You should segment the video based on the main topics and subtopics.
        - You should provide a concise summary of each fragment.
        - You should provide the timestamp of the start of each fragment.

        Please follow the format below:

        Transcript format:
        ---
        <MM:SS>
        <Transcript>
        ---

        Output format & requirements:
        - You can output the summary and your thinking process first before the timestamps.
        - But You should output the timestamp of the start of each fragment following MM:SS in the last line, separated by comma ','.
        - You should strictly follow the timestamp in the transcript. DO NOT MAKE UP ANY TIMESTAMP.
        - Remember only the timestamp is required.
        - Remember to output the timestamps in ascending order.
        - Remember to output the timestamps in the format MM:SS.
        - Remember to output the timestamps in the last line.

        Following is the transcript:
        <Transcript>
            """
            )
            + get_srt_text(srts)
            + "</Transcript>"
        )

    def parse_result(self, result: str, srts: list[Srt]):
        timestamp_line = result.strip().split("\n")[-1]
        ret = []
        for line in timestamp_line.split(","):
            mm, ss = map(int, line.strip().split(":"))
            ret.append(mm * 60 + ss)
        return ret


class LLMFrag(Model):
    def __init__(
        self,
        k: None | int = None,
        prompt: Literal["BP", "AP"] = "AP",
        model: Literal["gpt-4o"] = "gpt-4o",
    ):
        self.model = model
        self.k = k
        prompt_map = {
            "BP": BPPromptTemplate,
            "AP": APPromptTemplate,
        }
        if prompt in prompt_map:
            self.prompt_template = prompt_map[prompt]()
        else:
            raise ValueError(f"Unknown prompt: {prompt}")

    @property
    def requires_training(self):
        return False

    def fit(self, train_data, val_data=None):
        pass

    def _predict_one(self, video: Video):
        prompt = self.prompt_template.get_prompt(self.k, video.srt)
        result = call_llm(prompt)
        try:
            ret = self.prompt_template.parse_result(result, video.srt)
        except ValueError:
            print(f"Failed to parse result: {result}")
            raise
        ret = sorted(ret)
        if ret[0] != 0:
            ret = [0] + ret
        if ret[-1] != video.duration:
            ret = ret + [video.duration]
        return ret


register_model_runner(
    command_name="llm",
    model_cls=LLMFrag,
    model_param_build=lambda kwargs: dict(
        k=kwargs["k"],
        prompt_type=kwargs["prompt_type"],
        model=kwargs["model"],
    ),
    additional_options=[
        click.option(
            "--k",
            type=int,
            default=None,
            help="Number of fragments to detect. If not provided, the model will determine the number of fragments.",
        ),
        click.option(
            "--model",
            type=click.Choice(["gpt-4o"], case_sensitive=False),
            default="gpt-4o",
            help="LLM model to use for prediction.",
        ),
        click.option(
            "--prompt-type",
            type=click.Choice(["AP", "BP"], case_sensitive=False),
            default="AP",
            help="Prompt type to use for prediction. 'AP' for Analytical Prompt, 'BP' for Basic Prompt.",
        ),
    ],
)
