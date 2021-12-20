# Copyright 2021 Huawei Technologies Co., Ltd
"""Device adapter for ModelArts"""

from .config import config
if config.enable_modelarts:
    from .moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
else:
    from .local_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

__all__ = [
    'get_device_id', 'get_device_num', 'get_job_id', 'get_rank_id'
]
