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

def batched_data(data, batch_size):
    batch = []
    for cur in data:
        batch.append(cur)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
