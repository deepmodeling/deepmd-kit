import logging
import math
from typing import List
from typing import Tuple

import numpy as np

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import default_tf_session_config
from deepmd.env import op_module
from deepmd.env import tf

# from paddle.utils import cpp_extension
# op_module = cpp_extension.load(
#     name="custom_op_paddle2",
#     sources=["/workspace/hesensen/deepmd-kit/source/op/paddle/neighbor_stat.cc"],
#     extra_include_paths=["/workspace/hesensen/deepmd-kit/source/lib/include/","/usr/local/cuda/targets/x86_64-linux/include/", "/workspace/hesensen/deepmd-kit/source/op"],
#     # extra_library_paths=["../build/lib/", "/usr/local/cuda/lib64"],
#     verbose=True,
# )
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.parallel_op import ParallelOp

log = logging.getLogger(__name__)


class NeighborStat:
    """Class for getting training data information.

    It loads data from DeepmdData object, and measures the data info, including neareest nbor distance between atoms, max nbor size of atoms and the output data range of the environment matrix.

    Parameters
    ----------
    ntypes
            The num of atom types
    rcut
            The cut-off radius
    one_type : bool, optional, default=False
        Treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        one_type: bool = False,
    ) -> None:
        """Constructor."""
        self.rcut = rcut
        self.ntypes = ntypes
        self.one_type = one_type
        # sub_graph = tf.Graph()

        # def builder():
        #     place_holders = {}
        #     for ii in ["coord", "box"]:
        #         place_holders[ii] = tf.placeholder(
        #             GLOBAL_NP_FLOAT_PRECISION, [None, None], name="t_" + ii
        #         )
        #     place_holders["type"] = tf.placeholder(
        #         tf.int32, [None, None], name="t_type"
        #     )
        #     place_holders["natoms_vec"] = tf.placeholder(
        #         tf.int32, [self.ntypes + 2], name="t_natoms"
        #     )
        #     place_holders["default_mesh"] = tf.placeholder(
        #         tf.int32, [None], name="t_mesh"
        #     )
        #     t_type = place_holders["type"]
        #     t_natoms = place_holders["natoms_vec"]
        #     if self.one_type:
        #         # all types = 0, natoms_vec = [natoms, natoms, natoms]
        #         t_type = tf.clip_by_value(t_type, -1, 0)
        #         t_natoms = tf.tile(t_natoms[0:1], [3])
        #     _max_nbor_size, _min_nbor_dist = op_module.neighbor_stat( # 这里只计算一次
        #         place_holders["coord"],
        #         t_type,
        #         t_natoms,
        #         place_holders["box"],
        #         place_holders["default_mesh"],
        #         rcut=self.rcut,
        #     )
        #     place_holders["dir"] = tf.placeholder(tf.string)
        #     return place_holders, (_max_nbor_size, _min_nbor_dist, place_holders["dir"])

        # with sub_graph.as_default():
        #     self.p = ParallelOp(builder, config=default_tf_session_config)

        # self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)

    def get_stat(self, data: DeepmdDataSystem) -> Tuple[float, List[int]]:
        """Get the data statistics of the training data, including nearest nbor distance between atoms, max nbor size of atoms.

        Parameters
        ----------
        data
            Class for manipulating many data systems. It is implemented with the help of DeepmdData.

        Returns
        -------
        min_nbor_dist
            The nearest distance between neighbor atoms
        max_nbor_size
            A list with ntypes integers, denotes the actual achieved max sel
        """
        self.min_nbor_dist = 100.0
        self.max_nbor_size = [0]
        if not self.one_type:
            self.max_nbor_size *= self.ntypes

        # def feed():
        #     for ii in range(len(data.system_dirs)):
        #         for jj in data.data_systems[ii].dirs:
        #             data_set = data.data_systems[ii]._load_set(jj)
        #             for kk in range(np.array(data_set["type"]).shape[0]):
        #                 ret = {
        #                     "coord": np.array(data_set["coord"])[kk].reshape(
        #                         [-1, data.natoms[ii] * 3]
        #                     ), # (1, 576)
        #                     "type": np.array(data_set["type"])[kk].reshape(
        #                         [-1, data.natoms[ii]]
        #                     ), # (1, 192)
        #                     "natoms_vec": np.array(data.natoms_vec[ii]), # (4,)
        #                     "box": np.array(data_set["box"])[kk].reshape([-1, 9]), # (1, 9)
        #                     "default_mesh": np.array(data.default_mesh[ii]), # (6,)
        #                     "dir": str(jj), # ../data/data_0/set.xxx
        #                 }
        #                 print(str(jj))
        #                 print("coord", ret["coord"].shape, ret["coord"].dtype)
        #                 print("type", ret["type"].shape, ret["type"].dtype)
        #                 print("natoms_vec", ret["natoms_vec"].shape, ret["natoms_vec"].dtype)
        #                 print("box", ret["box"].shape, ret["box"].dtype)
        #                 print("default_mesh", ret["default_mesh"].shape, ret["default_mesh"].dtype)
        #                 # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/coord.npy", ret["coord"])
        #                 # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/type.npy", ret["type"])
        #                 # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/natoms_vec.npy", ret["natoms_vec"])
        #                 # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/box.npy", ret["box"])
        #                 # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/default_mesh.npy", ret["default_mesh"])
        #                 yield ret
        import paddle

        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]._load_set(jj)
                for kk in range(np.array(data_set["type"]).shape[0]):
                    coord = np.array(data_set["coord"])[kk].reshape(
                        [-1, data.natoms[ii] * 3]
                    )
                    coord = paddle.to_tensor(
                        coord, dtype="float32", place="cpu"
                    )  # [1, 576]

                    _type = np.array(data_set["type"])[kk].reshape(
                        [-1, data.natoms[ii]]
                    )
                    _type = paddle.to_tensor(
                        _type, dtype="int32", place="cpu"
                    )  # [1, 192]

                    natoms_vec = np.array(data.natoms_vec[ii])
                    natoms_vec = paddle.to_tensor(
                        natoms_vec, dtype="int64", place="cpu"
                    )  # [4]

                    box = np.array(data_set["box"])[kk].reshape([-1, 9])
                    box = paddle.to_tensor(box, dtype="float32", place="cpu")  # [1, 9]

                    default_mesh = np.array(data.default_mesh[ii])
                    default_mesh = paddle.to_tensor(
                        default_mesh, dtype="int32", place="cpu"
                    )  # [6]

                    rcut = self.rcut
                    mn, dt = op_module.neighbor_stat(
                        coord,
                        _type,
                        natoms_vec,
                        box,
                        default_mesh,
                        rcut,
                    )
                    if dt.size != 0:
                        dt = paddle.min(dt).item()
                    else:
                        dt = self.rcut
                        log.warning(
                            "Atoms with no neighbors found in %s. Please make sure it's what you expected."
                            % jj
                        )
                    if dt < self.min_nbor_dist:
                        if math.isclose(dt, 0.0, rel_tol=1e-6):
                            # it's unexpected that the distance between two atoms is zero
                            # zero distance will cause nan (#874)
                            raise RuntimeError(
                                "Some atoms are overlapping in %s. Please check your"
                                " training data to remove duplicated atoms." % jj
                            )
                        self.min_nbor_dist = dt
                    var = paddle.max(mn, axis=0).numpy()
                    self.max_nbor_size = np.maximum(var, self.max_nbor_size)

        # for mn, dt, jj in self.p.generate(self.sub_sess, feed()): # _max_nbor_size, _min_nbor_dist, dir
        #     # print(mn.shape, dt.shape, jj)
        #     # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/max_nbor_size.npy", mn)
        #     # np.save("/workspace/hesensen/deepmd-kit/cuda_ext/min_nbor_dist.npy", dt)
        #     if dt.size != 0:
        #         dt = np.min(dt)
        #     else:
        #         dt = self.rcut
        #         log.warning(
        #             "Atoms with no neighbors found in %s. Please make sure it's what you expected."
        #             % jj
        #         )
        #     if dt < self.min_nbor_dist:
        #         if math.isclose(dt, 0.0, rel_tol=1e-6):
        #             # it's unexpected that the distance between two atoms is zero
        #             # zero distance will cause nan (#874)
        #             raise RuntimeError(
        #                 "Some atoms are overlapping in %s. Please check your"
        #                 " training data to remove duplicated atoms." % jj
        #             )
        #         self.min_nbor_dist = dt
        #     var = np.max(mn, axis=0)
        #     self.max_nbor_size = np.maximum(var, self.max_nbor_size)

        log.info("training data with min nbor dist: " + str(self.min_nbor_dist))
        log.info("training data with max nbor size: " + str(self.max_nbor_size))
        return self.min_nbor_dist, self.max_nbor_size
