from dataclasses import dataclass
from functools import partial
from typing import Iterable

import click
import numpy as np
from scipy.spatial.distance import cosine

from kpi.cli import register_model_runner
from kpi.models.base_model import Model
from kpi.utils.preprocessing import SEN_MODEL, WORD_MODEL
from kpi.utils.video import Video


@dataclass
class Window:
    text: str
    srts: list[dict]

    @property
    def start(self):
        return self.srts[0]["start"]

    def _get_final_cover_ratio(self):
        final_text = self.srts[-1]["text"]
        for i in range(len(final_text), 0, -1):
            if self.text.endswith(final_text[:i]):
                return i
        assert False, f"Cannot match {final_text} in {self.text}"

    @property
    def end(self):
        fianl_start = self.srts[-1]["start"]
        final_duration = self.srts[-1]["duration"]
        cover_ratio = self._get_final_cover_ratio()
        return fianl_start + final_duration * cover_ratio


def _get_srt_window_helper(srt, window_size, window_step):
    ret = []
    for i in range(0, len(srt), window_step):
        end = min(i + window_size, len(srt))
        srts = srt[i:end]
        ret.append(Window(" ".join([x["text"] for x in srt[i:end]]), srts))
    return ret


def _get_text_window_helper(srt, window_size, window_step):
    full_text = []
    srt_idx = []
    for i, s in enumerate(srt):
        words = s["text"].split(" ")
        full_text += words
        srt_idx.extend([i] * len(words))
    assert len(full_text) == len(srt_idx)
    ret = []
    for i in range(0, len(full_text), window_step):
        end = min(i + window_size, len(full_text))
        srt_idx_window = sorted(list(set(srt_idx[i:end])))
        text = " ".join(full_text[i:end])
        srts = [srt[i] for i in srt_idx_window]
        ret.append(Window(text, srts))
    return ret


def _cosine_sim(vec):
    ret = []
    for i in range(len(vec) - 1):
        ret.append(1 - cosine(vec[i], vec[i + 1]))
    return ret


def extract_word_embed(wins: list[Window], model) -> np.ndarray:
    texts = [w.text for w in wins]
    docs = model.pipe(texts, batch_size=10240)
    ret = []
    for doc in docs:
        chunks = list(doc.noun_chunks)
        if not chunks:
            ret.append(doc.vector.get())
        else:
            ret.append(np.sum([x.vector.get() for x in chunks], axis=0))

    return np.array(ret)


def extract_sen_embed(wins: list[Window], model):
    texts = [w.text for w in wins]
    return model.encode(texts)


class TextTiling(Model):
    def __init__(
        self,
        policy: str,
        value: int,
        window_type: str,
        window_size: int,
        window_step: int,
        feature_type: str,
    ):
        super().__init__()
        self.policy = policy
        self.value = value
        self.window_type = window_type
        self.window_size = window_size
        self.window_step = window_step
        self.feature_type = feature_type
        if self.feature_type == self.FEATURE_TYPE_SEN:
            self._model = SEN_MODEL
        elif self.feature_type == self.FEATURE_TYPE_WORD:
            self._model = WORD_MODEL
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    FEATURE_TYPE_WORD = "word_embed"
    FEATURE_TYPE_SEN = "sen_embed"
    WINDOW_TYPE_SRT = "srt"
    WINDOW_TYPE_TEXT = "text"
    POLICY_THRESHOLD = "threshold"
    POLICY_TOP_K = "top_k"

    @property
    def model_name(self):
        return "TextTiling"

    def get_window_vec(self, windows):
        func = self._get_window_vec_by_type(self.feature_type)
        return func(windows, self._model)

    def _get_window_vec_by_type(self, feature_type):
        if feature_type == self.FEATURE_TYPE_WORD:
            func = extract_word_embed
        elif self.feature_type == self.FEATURE_TYPE_SEN:
            func = extract_sen_embed
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        return func

    def get_windows(self, srt):
        func = self._get_windows_by_type(
            self.window_type, self.window_size, self.window_step
        )
        return func(srt)

    def get_sim(self, vec):
        return _cosine_sim(vec)

    def _get_windows_by_type(self, window_type, window_size, window_step):
        if window_type == self.WINDOW_TYPE_SRT:
            func = partial(
                _get_srt_window_helper,
                window_size=window_size,
                window_step=window_step,
            )
        elif window_type == self.WINDOW_TYPE_TEXT:
            func = partial(
                _get_text_window_helper,
                window_size=window_size,
                window_step=window_step,
            )
        else:
            raise ValueError(f"Unknown window_type: {window_type}")
        return func

    @property
    def requires_training(self):
        return False

    def fit(self, train_data, val_data=None):
        pass

    @property
    def _if_print_progress(self):
        return True

    def _predict_one(self, video: Video):
        windows: Iterable[Window] = self.get_windows(video.srt)
        vec = self.get_window_vec(windows)
        sim = self.get_sim(vec)
        assert len(sim) == len(vec) - 1
        depths = self.get_depths(sim)
        ret = self.get_boundaries(depths, windows)
        return sorted(ret)

    def get_depths(self, sim):
        depths = [0 for _ in sim]

        for idx in range(1, len(sim) - 1):
            lpeak = sim[idx]
            for score in sim[idx::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = sim[idx]
            for score in sim[idx:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depths[idx] = lpeak + rpeak - 2 * sim[idx]
        return depths

    def get_boundaries(self, depths, windows):
        policy = self.policy
        value = self.value
        if policy == self.POLICY_THRESHOLD:
            mean = np.mean(depths)
            stdev = np.std(depths)
            thr = (mean - stdev) * self.value
            return [i.end for i, s in zip(windows, depths) if s > thr]
        elif policy == self.POLICY_TOP_K:
            sorted_depths = sorted(
                zip(windows, depths), key=lambda x: x[-1], reverse=True
            )
            return [i.end for i, _ in sorted_depths[:value]]
        else:
            raise ValueError(f"Unknown policy: {policy}")


register_model_runner(
    command_name="text_tiling",
    model_cls=TextTiling,
    model_param_build=lambda kwargs: dict(
        policy=kwargs["policy"],
        value=kwargs["k"],
        window_type=kwargs["win_type"],
        window_size=kwargs["win_size"],
        window_step=kwargs["win_step"],
        feature_type=kwargs["feat_type"],
    ),
    additional_options=[
        click.option(
            "--policy",
            type=click.Choice([TextTiling.POLICY_THRESHOLD, TextTiling.POLICY_TOP_K]),
            default=TextTiling.POLICY_TOP_K,
            help="Policy for boundary detection.",
        ),
        click.option(
            "--k",
            type=int,
            required=True,
            help="Value for the policy (threshold or top-k).",
        ),
        click.option(
            "--win_type",
            type=click.Choice(
                [TextTiling.WINDOW_TYPE_SRT, TextTiling.WINDOW_TYPE_TEXT]
            ),
            default=TextTiling.WINDOW_TYPE_TEXT,
            help="Type of window to use for text tiling.",
        ),
        click.option(
            "--win_size",
            type=int,
            required=True,
            help="Size of the window in terms of number of SRT entries or words.",
        ),
        click.option(
            "--win_step",
            type=int,
            required=True,
            help="Step size for moving the window.",
        ),
        click.option(
            "--feat_type",
            type=click.Choice(
                [TextTiling.FEATURE_TYPE_WORD, TextTiling.FEATURE_TYPE_SEN]
            ),
            default=TextTiling.FEATURE_TYPE_WORD,
            help="Type of feature to use for embedding.",
        ),
    ],
)
