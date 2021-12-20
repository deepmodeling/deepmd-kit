"""postprocess."""
import os
import numpy as np
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def cal_acc():
    """
    Calculate the accuracy of inference results.
    """
    result_path0 = os.path.join(config.post_result_path, "h2o_0.bin")
    result_path1 = os.path.join(config.post_result_path, "h2o_1.bin")
    energy = np.fromfile(result_path0, np.float32).reshape(1,)
    atom_ener = np.fromfile(result_path1, np.float32).reshape(192,)
    print('energy:', energy)
    print('atom_energy:', atom_ener)

    baseline = np.load(config.baseline_path)
    ae = baseline['e']

    if not np.mean((ae - atom_ener.reshape(-1,)) ** 2) < 3e-6:
        raise ValueError("Failed to varify atom_ener")

    print('successful')


if __name__ == '__main__':
    cal_acc()
