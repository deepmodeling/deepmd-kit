from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

import torch
import numpy as np

GLOBAL_PT_FLOAT_PRECISION = torch.float32 if GLOBAL_NP_FLOAT_PRECISION == np.float32 else torch.float64

EMBEDDING_NET_PATTERN = str(
    r"filter_type_(\d+)/(matrix)_(\d+)_(\d+)|"
    r"filter_type_(\d+)/(bias)_(\d+)_(\d+)|"
    r"filter_type_(\d+)/(idt)_(\d+)_(\d+)|"
    r"filter_type_(all)/(matrix)_(\d+)_(\d+)_(\d+)|"
    r"filter_type_(all)/(matrix)_(\d+)_(\d+)|"
    r"filter_type_(all)/(matrix)_(\d+)|"
    r"filter_type_(all)/(bias)_(\d+)_(\d+)_(\d+)|"
    r"filter_type_(all)/(bias)_(\d+)_(\d+)|"
    r"filter_type_(all)/(bias)_(\d+)|"
    r"filter_type_(all)/(idt)_(\d+)_(\d+)_(\d+)|"
    r"filter_type_(all)/(idt)_(\d+)_(\d+)|"
    r"filter_type_(all)/(idt)_(\d+)|"
)[:-1]