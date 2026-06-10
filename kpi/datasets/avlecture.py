import os
import glob
import srt
from functools import cache
from kpi.datasets.base_dataset import Dataset
from kpi.utils.video import Srt


def av_srt_reader(fn: str) -> list[Srt]:
    ret: list[Srt] = []
    with open(fn) as f:
        raw_srts = srt.parse(f)
        for s in raw_srts:
            if not s.content:
                continue
            if s.end.seconds - s.start.seconds < 1:
                continue
            ret.append(Srt(s.start.seconds, s.end.seconds, s.content))
        return ret


class AVLecture(Dataset):
    def __init__(self, fn_root: str):
        print("Loading dataset: AVLecture")

        self.fn_root = fn_root
        video_fns = glob.glob(os.path.join(f"{fn_root}/*/segmentation/videos/*.mp4"))

        course_ids = [fn.split("/")[-4] for fn in video_fns]
        video_ids = [fn.split("/")[-1].replace(".mp4", "") for fn in video_fns]
        assert len(video_fns) == len(course_ids) == len(video_ids)
        srt_fns = [
            f"{fn_root}/{course_ids[i]}/segmentation/subtitles/{video_ids[i]}.srt"
            for i in range(len(video_ids))
        ]
        for fn in srt_fns:
            if not os.path.exists(fn):
                raise FileNotFoundError(f"SRT file not found: {fn}")

        frags = [
            self._get_frags(course_ids[i], video_ids[i]) for i in range(len(video_ids))
        ]

        super().__init__(
            video_fns,
            srt_fns,
            frags,
            self._get_srt_reader(),
            print_info=True,
        )

    def _get_srt_reader(self):
        return av_srt_reader

    @cache
    def _get_course_info(self, course_id) -> list[tuple[str, float, float]]:
        segment_fn = f"{self.fn_root}/{course_id}/segmentation/segments_ts.txt"
        ret = []
        with open(segment_fn) as f:
            for line in f:
                line = line.strip()
                sep = "@@"
                _title, start, end, video_id = line.split(sep)
                start = float(start)
                end = float(end)
                ret.append((video_id, start, end))
        return ret

    def _get_frags(self, course_id, video_id) -> list[float]:
        course_info = self._get_course_info(course_id)
        ret = []
        for line in course_info:
            if line[0] == video_id:
                if len(ret) > 0:
                    assert ret[-1] == line[1]
                    ret.append(line[2])
                else:
                    ret.append(line[1])
                    ret.append(line[2])
        return ret
