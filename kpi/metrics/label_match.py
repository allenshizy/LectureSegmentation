from scipy.optimize import linear_sum_assignment


def get_pred_match_label(pred, gt):
    g = [[0 for __ in range(len(gt) - 1)] for _ in range(len(pred) - 1)]
    for s_idx in range(len(pred) - 1):
        pred_s = pred[s_idx]
        pred_e = pred[s_idx + 1]
        for g_idx in range(len(gt) - 1):
            gt_s = gt[g_idx]
            gt_e = gt[g_idx + 1]
            g[s_idx][g_idx] = max(min(pred_e, gt_e) - max(pred_s, gt_s), 0)
    pred_ind, gt_ind = linear_sum_assignment(g, maximize=True)
    labels = [-1] * (len(pred) - 1)
    for p, g in zip(pred_ind, gt_ind):
        labels[p] = g
    return labels


def get_label_list(frags: list[float], fps: int, labels=None):
    frags_i = list(map(round, frags))
    if labels:
        if len(labels) != len(frags_i) - 1:
            raise ValueError(f"Wrong labels length: {len(labels)}, {len(frags_i) - 1}")
    else:
        labels = list(range(len(frags_i) - 1))
    assert frags_i[0] == 0, f"Fragment should start with 0, {frags_i}"
    ret = [0] * (int(frags[-1]) * fps)
    for i, x in enumerate(frags_i[1:], 1):
        new_pos = int(x * fps)
        last_pos = int(frags_i[i - 1] * fps)
        ret[last_pos:new_pos] = [labels[i - 1]] * (new_pos - last_pos)
    return ret


assert get_label_list([0, 5, 10], 10) == [*([0] * (5 - 0) * 10), *([1] * (10 - 5) * 10)]
assert len(get_label_list([0, 440, 631, 1222, 1390, 2104, 2587, 2967], 10)) == 29670
assert get_label_list([0, 1, 3, 6], 1) == [0, 1, 1, 2, 2, 2]
assert get_label_list([0, 1, 3, 6], 1, [-1, 0, 1]) == [-1, 0, 0, 1, 1, 1]
