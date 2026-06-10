from kpi.metrics import Metric, cmp_higher_is_better
from kpi.metrics.label_match import get_label_list, get_pred_match_label


class MoF(Metric):
    def __init__(self, fps=10, matching=False):
        super().__init__(cmp=cmp_higher_is_better)
        self.fps = fps
        self.matching = matching

    def _cal_single(self, pred, gt):
        if not self.matching:
            pred_list = get_label_list(pred, self.fps)
        else:
            labels = get_pred_match_label(pred, gt)
            pred_list = get_label_list(pred, self.fps, labels)
        gt_list = get_label_list(gt, self.fps)
        assert len(pred_list) == len(gt_list), (pred, gt)
        ret = 0
        for i in range(len(pred_list)):
            if pred_list[i] == gt_list[i]:
                ret += 1
        return ret / len(pred_list)


assert (tmp := MoF().calculate([[0, 3, 5, 6]], [[0, 3, 4, 6]])) == 5 / 6, tmp
assert (
    tmp := MoF(matching=True).calculate([[0, 1, 3, 6, 10]], [[0, 3, 6, 10]])
) == 9 / 10, tmp
