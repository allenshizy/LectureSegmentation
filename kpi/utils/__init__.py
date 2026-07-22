def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from kpi.utils.text_label_dataset import (
    TextLabelSequenceDataset,
    filter_internal_boundaries,
    map_frags_to_sentence_labels,
    text_label_collate_fn,
    video_to_text_and_labels,
)
from kpi.utils.labels_to_boundary_times import (
    labels_to_boundary_times,
    video_labels_to_boundary_times,
)
from kpi.utils.visualization import (
    plot_probs_by_sentence,
    plot_probs_by_time,
)

def batched_data(data, batch_size):
    batch = []
    for cur in data:
        batch.append(cur)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
