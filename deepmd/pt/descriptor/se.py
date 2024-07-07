import re
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.env import (
    EMBEDDING_NET_PATTERN,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

import torch
#return "/share/home/wangyan/project/pytorch/torch"