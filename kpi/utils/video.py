import ffmpeg
from math import ceil
from functools import cached_property


def get_duration(video_fn):
    return float(ffmpeg.probe(video_fn)["format"]["duration"])


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
