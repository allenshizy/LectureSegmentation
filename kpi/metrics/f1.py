from kpi.metrics import Metric, cmp_higher_is_better


def _check_equal(pred, gt, threshold):
    return abs(pred - gt) <= threshold


class Recall(Metric):
    def __init__(self, threshold=0.5):
        super().__init__(cmp=cmp_higher_is_better)
        self.threshold = threshold

    def _cal_single(self, pred, gt):
        pred = pred[1:-1]
        gt = gt[1:-1]
        if len(gt) == 0:
            return 0
        vis_gt = {}
        cnt = 0
        for p in pred:
            for i, g in enumerate(gt):
                if i not in vis_gt and _check_equal(p, g, self.threshold):
                    vis_gt[i] = 1
                    cnt += 1
                    break
        return cnt / len(gt)


class Precision(Metric):
    def __init__(self, threshold=0.5):
        super().__init__(cmp=cmp_higher_is_better)
        self.threshold = threshold

    def _cal_single(self, pred, gt):
        # pred and gt should start with 0 and end with the same value
        pred = pred[1:-1]
        gt = gt[1:-1]
        if len(pred) == 0:
            return 0
        vis_gt = {}
        cnt = 0
        for p in pred:
            for i, g in enumerate(gt):
                if i not in vis_gt and _check_equal(p, g, self.threshold):
                    vis_gt[i] = 1
                    cnt += 1
                    break
        return cnt / len(pred)


class F1(Metric):
    def __init__(self, threshold=0.5):
        super().__init__(cmp=cmp_higher_is_better)
        self.threshold = threshold

    def _cal_single(self, pred, gt):
        # pred and gt should start with 0 and end with the same value
        pred = pred[1:-1]
        gt = gt[1:-1]
        vis_gt = {}
        cnt = 0
        for p in pred:
            for i, g in enumerate(gt):
                if i not in vis_gt and _check_equal(p, g, self.threshold):
                    vis_gt[i] = 1
                    cnt += 1
                    break
        precision = cnt / len(pred) if pred else 0
        recall = cnt / len(gt)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)


assert (tmp := Recall().calculate([[0, 1, 2, 3, 4, 5]], [[0, 5]])) == 0, tmp
assert (tmp := Recall(2).calculate([[0, 1, 2, 3, 4, 5]], [[0, 3, 5]])) == 1, tmp
