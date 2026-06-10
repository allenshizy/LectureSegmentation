from kpi.metrics import Metric, cmp_higher_is_better
from kpi.metrics.label_match import get_label_list, get_pred_match_label


class IoU(Metric):
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

        pred_maps = dict()
        gt_maps = dict()
        for i in range(len(pred_list)):
            if pred_list[i] not in pred_maps:
                pred_maps[pred_list[i]] = set()
            pred_maps[pred_list[i]].add(i)

            if gt_list[i] not in gt_maps:
                gt_maps[gt_list[i]] = set()
            gt_maps[gt_list[i]].add(i)

        max_id = max(max(pred_list), max(gt_list))
        ret = 0
        for i in range(max_id + 1):
            if i not in pred_maps:
                pred_maps[i] = set()
            if i not in gt_maps:
                gt_maps[i] = set()

            inter = pred_maps[i].intersection(gt_maps[i])
            union = pred_maps[i].union(gt_maps[i])
            if union:
                ret += len(inter) / len(union)
        return ret / (max_id + 1)


assert (tmp := IoU().calculate([[0, 3, 5, 6]], [[0, 3, 4, 6]])) == (
    1 + (1 / 2) + (1 / 2)
) / 3, tmp
