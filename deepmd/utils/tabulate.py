import re
import math
import logging
import numpy as np
from typing import Callable
from typing import Tuple, List
from deepmd.env import tf
from deepmd.env import op_module
from deepmd.common import ACTIVATION_FN_DICT
from deepmd.utils.sess import run_sess
from deepmd.utils.graph import get_tensor_by_name_from_graph, load_graph_def 
from deepmd.utils.graph import get_embedding_net_nodes_from_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

log = logging.getLogger(__name__)

class DPTabulate():
    """
    Class for tabulation.

    Compress a model, which including tabulating the embedding-net. 
    The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first table takes the stride(parameter) as it\'s uniform stride, while the second table takes 10 * stride as it\s uniform stride 
    The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table\'s upper boundary(upper) to the extrapolate(parameter) * upper.

    Parameters
    ----------
    model_file
            The frozen model
    type_one_side
            Try to build N_types tables. Otherwise, building N_types^2 tables
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    activation_function
            The activation function in the embedding net. Supported options are {"tanh","gelu"} in common.ACTIVATION_FN_DICT.
    suffix : str, optional
            The suffix of the scope
    """
    def __init__(self,
                 model_file : str,
                 type_one_side : bool = False,
                 exclude_types : List[List[int]] = [],
                 activation_fn : Callable[[tf.Tensor], tf.Tensor] = tf.nn.tanh,
                 suffix : str = "",
                 ) -> None:
        """
        Constructor
        """

        self.model_file = model_file
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
        self.suffix = suffix
        if self.type_one_side and len(self.exclude_types) != 0:
            raise RuntimeError('"type_one_side" is not compatible with "exclude_types"')
        
        # functype
        if activation_fn == ACTIVATION_FN_DICT["tanh"]:
            self.functype = 1
        elif activation_fn == ACTIVATION_FN_DICT["gelu"]:
            self.functype = 2
        else:
            raise RuntimeError("Unknown actication function type!")
        self.activation_fn = activation_fn

        self.graph, self.graph_def = load_graph_def(self.model_file)
        self.sess = tf.Session(graph = self.graph)

        self.sub_graph, self.sub_graph_def = self._load_sub_graph()
        self.sub_sess = tf.Session(graph = self.sub_graph)

        try:
            self.sel_a = self.graph.get_operation_by_name('ProdEnvMatA').get_attr('sel_a')
            self.descrpt = self.graph.get_operation_by_name ('ProdEnvMatA')
        except Exception:
            self.sel_a = self.graph.get_operation_by_name('DescrptSeA').get_attr('sel_a')
            self.descrpt = self.graph.get_operation_by_name ('DescrptSeA')

        self.davg = get_tensor_by_name_from_graph(self.graph, f'descrpt_attr{self.suffix}/t_avg')
        self.dstd = get_tensor_by_name_from_graph(self.graph, f'descrpt_attr{self.suffix}/t_std')
        self.ntypes = get_tensor_by_name_from_graph(self.graph, 'descrpt_attr/ntypes')

        
        self.rcut = self.descrpt.get_attr('rcut_r')
        self.rcut_smth = self.descrpt.get_attr('rcut_r_smth')

        self.embedding_net_nodes = get_embedding_net_nodes_from_graph_def(self.graph_def, suffix=self.suffix)

        for tt in self.exclude_types:
            if (tt[0] not in range(self.ntypes)) or (tt[1] not in range(self.ntypes)):
                raise RuntimeError("exclude types" + str(tt) + " must within the number of atomic types " + str(self.ntypes) + "!")
        if (self.ntypes * self.ntypes - len(self.exclude_types) == 0):
            raise RuntimeError("empty embedding-net are not supported in model compression!")
        
        self.layer_size = len(self.embedding_net_nodes) // ((self.ntypes * self.ntypes - len(self.exclude_types)) * 2)
        self.table_size = self.ntypes * self.ntypes
        if type_one_side :
            self.layer_size = len(self.embedding_net_nodes) // (self.ntypes * 2)
            self.table_size = self.ntypes
        # self.value_type = self.embedding_net_nodes["filter_type_0/matrix_1_0"].dtype #"filter_type_0/matrix_1_0" must exit~
        # get trained variables
        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                self.data_type = type(item[0][0])
                self.last_layer_size = item.shape[1]
        # define tables
        self.data = {}


    def build(self, 
              min_nbor_dist : float,
              extrapolate : float, 
              stride0 : float, 
              stride1 : float) -> Tuple[int, int]:
        """
        Build the tables for model compression

        Parameters
        ----------
        min_nbor_dist
                The nearest distance between neighbor atoms
        extrapolate
                The scale of model extrapolation
        stride0
                The uniform stride of the first table
        stride1
                The uniform stride of the second table

        Returns
        ----------
        lower
                The lower boundary of environment matrix
        upper
                The upper boundary of environment matrix
        """
        # tabulate range [lower, upper] with stride0 'stride0'
        lower, upper = self._get_env_mat_range(min_nbor_dist)
        xx = np.arange(lower, upper, stride0, dtype = self.data_type)
        xx = np.append(xx, np.arange(upper, extrapolate * upper, stride1, dtype = self.data_type))
        xx = np.append(xx, np.array([extrapolate * upper], dtype = self.data_type))
        self.nspline = int((upper - lower) / stride0 + (extrapolate * upper - upper) / stride1)
        for ii in range(self.table_size):
            if self.type_one_side or (ii // self.ntypes, int(ii % self.ntypes)) not in self.exclude_types:
                vv, dd, d2 = self._make_data(xx, ii)
                if self.type_one_side:
                    net = "filter_-1_net_" + str(ii)
                else:
                    net = "filter_" + str(ii // self.ntypes) + "_net_" + str(int(ii % self.ntypes))
                self.data[net] = np.zeros([self.nspline, 6 * self.last_layer_size], dtype = self.data_type)
                # for jj in tqdm(range(self.nspline), desc = 'DEEPMD INFO    |-> deepmd.utils.tabulate\t\t\t' + net + ', tabulating'):
                for jj in range(self.nspline):
                    for kk in range(self.last_layer_size):
                        if jj < int((upper - lower) / stride0):
                            tt = stride0
                        else:
                            tt = stride1
                        hh = vv[jj + 1][kk] - vv[jj][kk]
                        self.data[net][jj][kk * 6 + 0] = vv[jj][kk]
                        self.data[net][jj][kk * 6 + 1] = dd[jj][kk]
                        self.data[net][jj][kk * 6 + 2] = 0.5 * d2[jj][kk]
                        self.data[net][jj][kk * 6 + 3] = (1 / (2 * tt * tt * tt)) * (20 * hh - (8 * dd[jj + 1][kk] + 12 * dd[jj][kk]) * tt - (3 * d2[jj][kk] - d2[jj + 1][kk]) * tt * tt)
                        self.data[net][jj][kk * 6 + 4] = (1 / (2 * tt * tt * tt * tt)) * (-30 * hh + (14 * dd[jj + 1][kk] + 16 * dd[jj][kk]) * tt + (3 * d2[jj][kk] - 2 * d2[jj + 1][kk]) * tt * tt)
                        self.data[net][jj][kk * 6 + 5] = (1 / (2 * tt * tt * tt * tt * tt)) * (12 * hh - 6 * (dd[jj + 1][kk] + dd[jj][kk]) * tt + (d2[jj + 1][kk] - d2[jj][kk]) * tt * tt)
        return lower, upper

    def _load_sub_graph(self):
        sub_graph_def = tf.GraphDef()
        with tf.Graph().as_default() as sub_graph:
            tf.import_graph_def(sub_graph_def, name = "")
        return sub_graph, sub_graph_def

    def _get_bias(self):
        bias = {}
        for layer in range(1, self.layer_size + 1):
            bias["layer_" + str(layer)] = []
            if self.type_one_side:
                for ii in range(0, self.ntypes):
                    node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/bias_{layer}_{ii}"]
                    tensor_value = np.frombuffer (node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
                    tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
                    bias["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
            else:
                for ii in range(0, self.ntypes * self.ntypes):
                    if (ii // self.ntypes, int(ii % self.ntypes)) not in self.exclude_types:
                        node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/bias_{layer}_{ii % self.ntypes}"]
                        tensor_value = np.frombuffer(node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
                        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
                        bias["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
                    else:
                        bias["layer_" + str(layer)].append(np.array([]))
        return bias

    def _get_matrix(self):
        matrix = {}
        for layer in range(1, self.layer_size + 1):
            matrix["layer_" + str(layer)] = []
            if self.type_one_side:
                for ii in range(0, self.ntypes):
                    node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/matrix_{layer}_{ii}"]
                    tensor_value = np.frombuffer (node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
                    tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
                    matrix["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
            else:
                for ii in range(0, self.ntypes * self.ntypes):
                    if (ii // self.ntypes, int(ii % self.ntypes)) not in self.exclude_types:
                        node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/matrix_{layer}_{ii % self.ntypes}"]
                        tensor_value = np.frombuffer(node.tensor_content, dtype = tf.as_dtype(node.dtype).as_numpy_dtype)
                        tensor_shape = tf.TensorShape(node.tensor_shape).as_list()
                        matrix["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
                    else:
                        matrix["layer_" + str(layer)].append(np.array([]))
        return matrix

    # one-by-one executions
    def _make_data(self, xx, idx):
        with self.sub_graph.as_default():
            with self.sub_sess.as_default():
                xx = tf.reshape(xx, [xx.size, -1])
                for layer in range(self.layer_size):
                    if layer == 0:
                        xbar = tf.matmul(
                            xx, self.matrix["layer_" + str(layer + 1)][idx]) + self.bias["layer_" + str(layer + 1)][idx]
                        yy = self._layer_0(
                            xx, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                        dy = op_module.unaggregated_dy_dx_s(
                            yy, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype))
                        dy2 = op_module.unaggregated_dy2_dx_s(
                            yy, dy, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype))
                    else:
                        ybar = tf.matmul(
                            yy, self.matrix["layer_" + str(layer + 1)][idx]) + self.bias["layer_" + str(layer + 1)][idx]
                        tt, zz = self._layer_1(
                            yy, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                        dz = op_module.unaggregated_dy_dx(
                            zz - tt, self.matrix["layer_" + str(layer + 1)][idx], dy, ybar, tf.constant(self.functype))
                        dy2 = op_module.unaggregated_dy2_dx(
                            zz - tt, self.matrix["layer_" + str(layer + 1)][idx], dy, dy2, ybar, tf.constant(self.functype))
                        dy = dz
                        yy = zz

                vv = zz.eval()
                dd = dy.eval()
                d2 = dy2.eval()
        return vv, dd, d2

    def _layer_0(self, x, w, b):
        return self.activation_fn(tf.matmul(x, w) + b)

    def _layer_1(self, x, w, b):
        t = tf.concat([x, x], axis=1)
        return t, self.activation_fn(tf.matmul(x, w) + b) + t

    def _save_data(self):
        for ii in range(self.ntypes * self.ntypes):
            net = "filter_" + str(ii // self.ntypes) + "_net_" + str(int(ii % self.ntypes))
            np.savetxt('data_' + str(ii), self.data[net])

    def _get_env_mat_range(self,
                           min_nbor_dist):
        lower = 100.0
        upper = -10.0
        sw    = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
        for ii in range(self.ntypes):
            if lower > -self.davg[ii][0] / self.dstd[ii][0]:
                lower = -self.davg[ii][0] / self.dstd[ii][0]
            if upper < ((1 / min_nbor_dist) * sw - self.davg[ii][0]) / self.dstd[ii][0]:
                upper = ((1 / min_nbor_dist) * sw - self.davg[ii][0]) / self.dstd[ii][0]
        log.info('training data with lower boundary: ' + str(lower))
        log.info('training data with upper boundary: ' + str(upper))
        return math.floor(lower), math.ceil(upper)

    def _spline5_switch(self,
                        xx,
                        rmin,
                        rmax):
        if xx < rmin:
            vv = 1
        elif xx < rmax:
            uu = (xx - rmin) / (rmax - rmin)
            vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1
        else:
            vv = 0
        return vv
