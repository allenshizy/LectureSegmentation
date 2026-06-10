from abc import ABC, abstractmethod


def cmp_higher_is_better(a, b):
    return a > b


def cmp_lower_is_better(a, b):
    return a < b


def cmp_abs_lower_is_better(a, b):
    return abs(a) < abs(b)


class Metric(ABC):
    def __init__(self, cmp):
        self.cmp = cmp

    def calculate(self, preds, gts):
        assert len(preds) == len(gts), (
            f"preds, gts should have the same length, {preds}, {gts}"
        )
        for i in range(len(preds)):
            preds[i] = list(map(round, preds[i]))
            gts[i] = list(map(round, gts[i]))
            assert preds[i][0] == gts[i][0] == 0, (
                f"pred, gt should start with 0, {preds[i]}, {gts[i]}"
            )
            assert preds[i][-1] == gts[i][-1], (
                f"pred, gt should end with the same value, {preds[i]}, {gts[i]}"
            )

        return sum(self._cal_single(pred, gt) for pred, gt in zip(preds, gts)) / len(
            preds
        )

    @abstractmethod
    def _cal_single(self, pred, gt):
        raise NotImplementedError

    def __repr__(self):
        configs = []
        for k in self.__dict__:
            if k == "cmp":
                continue
            configs.append(f"{k}={self.__dict__[k]}")
        return f"{self.__class__.__name__} ({', '.join(configs)})"

    def __call__(self, preds, gts):
        return self.calculate(preds, gts)




