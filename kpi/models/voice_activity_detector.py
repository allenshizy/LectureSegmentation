from kpi.models.base_model import Model
from kpi.utils.video import Video
import click
from kpi.cli import register_model_runner


class VoiceActivityDetector(Model):
    def __init__(self, detector="srt", k=2):
        self.detector = detector
        if detector == "srt":
            self.k = k
        else:
            raise ValueError("Unknown detector")
        super().__init__()

    @property
    def requires_training(self):
        return False

    def fit(self, train_data, val_data=None):
        pass

    @property
    def _if_print_progress(self):
        return True

    @property
    def model_name(self):
        return "VAD"

    def _predict_one(self, video: Video):
        if self.detector == "srt":
            silences = []
            for i in range(1, len(video.srt)):
                last_start = video.srt[i - 1]["start"]
                last_end = last_start + video.srt[i - 1]["duration"]
                cur_start = video.srt[i]["start"]
                silences.append({"start": last_end, "duration": cur_start - last_end})
            ret = self._get_k_frags(silences, self.k)
        else:
            raise ValueError("Unknown detector")
        return sorted([i for i in ret if i < video.duration])

    def _get_k_frags(self, silences, k):
        silences = sorted(silences, key=lambda x: x["duration"], reverse=True)
        return [x["start"] for x in silences[:k]]

register_model_runner(
    command_name="vad",
    model_cls=VoiceActivityDetector,
    model_param_build=lambda kwargs: dict(k=kwargs["k"]),
    additional_options=[
        click.option(
            "--k",
            type=int,
            required=True,
            help="Number of fragments to detect.",
        ),
    ],
)
