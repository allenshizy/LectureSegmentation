import shutil

import ffmpeg
from math import ceil
from functools import cached_property


def get_duration(video_fn):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg executable was not found on PATH. Install ffmpeg and make sure the "
            "ffmpeg command is available before running MITFLD training."
        )

    try:
        return float(ffmpeg.probe(video_fn)["format"]["duration"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable was not found when probing video duration. "
            "Install ffmpeg and add it to PATH."
        ) from exc


class Video:
    def __init__(self, video_fn, srt_fn, srt_reader):
        self.video_fn = video_fn
        self.srt_fn = srt_fn
        self.srt_reader = srt_reader

    @cached_property
    def duration(self):
        return ceil(get_duration(self.video_fn))

    @cached_property
    def srt(self):
        return self.srt_reader(self.srt_fn)


class Srt:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

    @property
    def duration(self):
        return self.end - self.start

    def __getitem__(self, name):
        return getattr(self, name)
