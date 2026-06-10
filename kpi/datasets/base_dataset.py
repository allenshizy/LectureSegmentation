import random
from typing import Callable, Iterable
from kpi.utils.video import Video, Srt


class Dataset:

    def __init__(
        self,
        video_fns: list[str],
        srt_fns: list[str],
        frags: list[list[float | int]],
        srt_reader: Callable[[str], Iterable[Srt]],
        print_info=True,
    ):
        self.video_fns = video_fns
        self.srt_fns = srt_fns
        self.srt_reader = srt_reader
        self.frags = frags
        self.videos = [
            Video(vfn, sfn, srt_reader)
            for vfn, sfn in zip(self.video_fns, self.srt_fns)
        ]

        EPS = 1
        for frag, v in zip(self.frags, self.videos):
            if frag[0] > EPS:
                frag.insert(0, 0)
            else:
                frag[0] = 0
            if v.duration - frag[-1] < 2 * EPS:
                frag[-1] = v.duration
            else:
                frag.append(v.duration)
            assert frag[-1] == v.duration, (frag, v.duration)

        frag_tot = sum(len(frag) - 1 for frag in self.frags)
        duration_tot = sum(v.duration for v in self.videos)

        if print_info:
            print(
                f"Loaded {len(self)} videos, {frag_tot} fragments, {duration_tot} seconds"
            )
            print(f"Average duration per video: {duration_tot / len(self):.2f} seconds")
            print(
                f"Average duration per fragment: {duration_tot / frag_tot:.2f} seconds"
            )
            print(f"Average frag per video: {frag_tot / len(self):.2f}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.frags[idx]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration
        else:
            cur = self[self._idx]
            self._idx += 1
            return cur

    def random_split(self, ratios, seed=2024):
        if sum(ratios) > 1:
            raise ValueError("Sum of ratios must be less than or equal to 1")
        random.seed(seed)
        indices = list(range(len(self)))
        random.shuffle(indices)
        for i in range(len(ratios)):
            start = int(sum(ratios[:i]) * len(indices))
            if i == len(ratios) - 1:
                end = len(indices)
            else:
                end = int(sum(ratios[: i + 1]) * len(indices))

            video_fns = [self.video_fns[i] for i in indices[start:end]]
            srt_fns = [self.srt_fns[i] for i in indices[start:end]]
            frags = [self.frags[i] for i in indices[start:end]]
            yield Dataset(
                video_fns,
                srt_fns,
                frags,
                self.srt_reader,
                print_info=False,
            )

    def _get_srt_reader(self) -> Callable[[str], list[Srt]]:
        raise NotImplementedError
