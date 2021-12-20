"""eval."""
import numpy as np
from mindspore import Tensor
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.network import Network
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=config.device_target)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    """
    export network model
    """
    net = Network()
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.to_float(mstype.float32)
    d_coord = Tensor(np.zeros([1, 1536, 3], np.float32))
    d_nlist = Tensor(np.zeros([1, 192, 138], np.int32))
    avg = Tensor(np.zeros([2, 552], np.float16))
    std = Tensor(np.zeros([2, 552], np.float16))
    atype = Tensor(np.zeros([192,], np.int32))
    nlist = Tensor(np.zeros([1, 192, 138], np.int32))

    batch_size = 1
    frames = []
    for i in range(batch_size):
        frames.append(i * 1536)
    frames = Tensor(np.array(frames, np.int32))
    input_arr = (d_coord, d_nlist, frames, avg, std, atype, nlist)
    export(net, *input_arr, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    model_export()
