# SPDX-License-Identifier: LGPL-3.0-or-later
from collections import (
    defaultdict,
)
from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np


def weighted_average(errors: List[Dict[str, Tuple[float, float]]]) -> Dict:
    """Compute wighted average of prediction errors (MAE or RMSE) for model.

    Parameters
    ----------
    errors : List[Dict[str, Tuple[float, float]]]
        List: the error of systems
        Dict: the error of quantities, name given by the key
        str: the name of the quantity, must starts with 'mae' or 'rmse'
        Tuple: (error, weight)

    Returns
    -------
    Dict
        weighted averages
    """
    sum_err = defaultdict(float)
    sum_siz = defaultdict(int)
    for err in errors:
        for kk, (ee, ss) in err.items():
            if kk.startswith("mae"):
                sum_err[kk] += ee * ss
            elif kk.startswith("rmse"):
                sum_err[kk] += ee * ee * ss
            else:
                raise RuntimeError("unknown error type")
            sum_siz[kk] += ss
    for kk in sum_err.keys():
        if kk.startswith("mae"):
            sum_err[kk] = sum_err[kk] / sum_siz[kk]
        elif kk.startswith("rmse"):
            sum_err[kk] = np.sqrt(sum_err[kk] / sum_siz[kk])
        else:
            raise RuntimeError("unknown error type")
    return sum_err
