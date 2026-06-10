import json
import orjson
import os
from kpi.datasets.base_dataset import Dataset
from kpi.utils.video import Srt


def mit_srt_reader(fn: str):
    return [
        Srt(x["start"], x["start"] + x["duration"], x["text"])
        for x in orjson.loads(
            open(fn).read(),
        )
    ]


class MITFLD(Dataset):

    FRAG_DIR = "frags"
    SRT_DIR = "transcripts"
    VIDEO_DIR = "videos"

    def __init__(self, fn_root: str):
        print("Loading MITFLD")
        srt_root = os.path.join(fn_root, self.SRT_DIR)
        frag_root = os.path.join(fn_root, self.FRAG_DIR)
        video_root = os.path.join(fn_root, self.VIDEO_DIR)
        video_fns = []
        srt_fns = []
        frag_fns = []
        with open(os.path.join(fn_root, "video_id_list.txt"), "r") as f:
            for line in f:
                vid = line.strip()
                video_fns += [os.path.join(video_root, f"{vid}.mp4")]
                srt_fns += [os.path.join(srt_root, f"{vid}.json")]
                frag_fns += [os.path.join(frag_root, f"{vid}.json")]
        frags = [json.load(open(fn)) for fn in frag_fns]
        super().__init__(
            video_fns,
            srt_fns,
            frags,
            self._get_srt_reader(),
        )

    def _get_srt_reader(self):
        return mit_srt_reader
