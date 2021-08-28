from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
import numpy as np


def weighted_average(
    errors: List[Dict[str, Tuple[float, float]]]
) -> Dict:
    """Compute wighted average of prediction errors for model.

    Parameters
    ----------
    errors : List[Dict[str, Tuple[float, float]]]
        List: the error of systems
        Dict: the error of quantities, name given by the key
        Tuple: (error, weight)

    Returns
    -------
    Dict
        weighted averages
    """
    sum_err = {}
    sum_siz = {}
    for err in errors:
        for kk, (ee, ss) in err.items():
            if kk in sum_err:
                sum_err[kk] += ee * ee * ss
                sum_siz[kk] += ss
            else :
                sum_err[kk] = ee * ee * ss
                sum_siz[kk] = ss
    for kk in sum_err.keys():
        sum_err[kk] = np.sqrt(sum_err[kk] / sum_siz[kk])
    return sum_err
