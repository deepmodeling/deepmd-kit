"""preprocess."""
import os
import numpy as np
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    pass

@moxing_wrapper(pre_process=modelarts_pre_process)
def generate_bin():
    """
    convert input data to numpy bin files
    """
    r = np.load(config.dataset_path)
    d_coord, d_nlist, avg, std, atype, nlist = r['d_coord'], r['d_nlist'], r['avg'], r['std'], r['atype'], r['nlist']
    batch_size = 1
    d_coord = np.reshape(d_coord.astype(np.float32), (1, -1, 3))
    frames = []
    for i in range(batch_size):
        frames.append(i * 1536)
    frames = np.array(frames).astype(np.int32)
    dir_list = ["00_d_coord", "01_d_nlist", "02_frames", "03_avg", "04_std", "05_atype", "06_nlist"]
    path_list = []
    for d in dir_list:
        p = os.path.join(config.pre_result_path, d)
        os.makedirs(p)
        path_list.append(p)

    file_name = "h2o.bin"
    array_list = [d_coord, d_nlist, frames, avg, std, atype, nlist]
    for i, f in enumerate(array_list):
        f.tofile(os.path.join(path_list[i], file_name))

    print("="*20, "export bin files finished", "="*20)

if __name__ == '__main__':
    generate_bin()
