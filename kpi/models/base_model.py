import sys
from abc import ABC, abstractmethod
from typing import Callable
from tqdm import tqdm
from kpi.utils.video import Video


class Model(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_data, val_data=None):
        raise NotImplementedError

    @abstractmethod
    def _predict_one(self, video: Video):
        if self.requires_training:
            raise RuntimeError("Model is not fitted")
        raise NotImplementedError

    @property
    def _if_print_progress(self) -> bool:
        return False

    def predict(self, videos: list[Video]):
        if self._if_print_progress:
            print(f"{self} predicting...", file=sys.stderr, flush=True)
            iter_X = tqdm(videos)
        else:
            iter_X = videos
        return [self._post_process(self._predict_one(x), x) for x in iter_X]

    def _post_process(self, ret, video):
        # handle padding
        ret = sorted([x for x in ret if 0 <= x <= video.duration])
        if not ret:
            ret = [0, video.duration]
        if ret[0] != 0:
            ret = [0] + ret
        if ret[-1] != video.duration:
            ret = ret + [video.duration]
        return ret

    @property
    def model_name(self):
        return self.__class__.__name__

    def __repr__(self):
        configs = []
        for k in self.__dict__:
            if k.startswith("_"):
                continue
            if isinstance(self.__dict__[k], Callable):
                configs.append(f"{k}={self.__dict__[k].__name__}")
                continue
            configs.append(f"{k}={self.__dict__[k]}")
        return f"{self.__class__.__name__} ({', '.join(configs)})"

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
