import os
import random
import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Garantisce risultati deterministici a scapito di una lieve riduzione di velocità
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
