import math
import logging
import numpy as np
import deepmd
from typing import Callable
from typing import Tuple, List
from scipy.special import comb
from deepmd.env import tf
from deepmd.env import op_module
from deepmd.common import ACTIVATION_FN_DICT
from deepmd.utils.graph import get_tensor_by_name_from_graph, load_graph_def 
from deepmd.utils.graph import get_embedding_net_nodes_from_graph_def
from deepmd.descriptor import Descriptor

log = logging.getLogger(__name__)

class DPTabulate():
    """
    Class for tabulation.

    Compress a model, which including tabulating the embedding-net. 
    The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first table takes the stride(parameter) as it\'s uniform stride, while the second table takes 10 * stride as it\s uniform stride 
    The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table\'s upper boundary(upper) to the extrapolate(parameter) * upper.

    Parameters
    ----------
    descrpt
            Descriptor of the original model
    neuron
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
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
                 descrpt : Descriptor,
                 neuron : List[int],
                 model_file : str,
                 type_one_side : bool = False,
                 exclude_types : List[List[int]] = [],
                 activation_fn : Callable[[tf.Tensor], tf.Tensor] = tf.nn.tanh,
                 suffix : str = "",
                 ) -> None:
        """
        Constructor
        """
        self.descrpt = descrpt
        self.neuron = neuron
        self.model_file = model_file
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
        self.suffix = suffix
        
        # functype
        if activation_fn == ACTIVATION_FN_DICT["tanh"]:
            self.functype = 1
        elif activation_fn == ACTIVATION_FN_DICT["gelu"]:
            self.functype = 2
        elif activation_fn == ACTIVATION_FN_DICT["relu"]:
            self.functype = 3
        elif activation_fn == ACTIVATION_FN_DICT["relu6"]:
            self.functype = 4
        elif activation_fn == ACTIVATION_FN_DICT["softplus"]:
            self.functype = 5
        elif activation_fn == ACTIVATION_FN_DICT["sigmoid"]:
            self.functype = 6
        else:
            raise RuntimeError("Unknown actication function type!")
        self.activation_fn = activation_fn

        self.graph, self.graph_def = load_graph_def(self.model_file)
        self.sess = tf.Session(graph = self.graph)

        self.sub_graph, self.sub_graph_def = self._load_sub_graph()
        self.sub_sess = tf.Session(graph = self.sub_graph)

        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            try:
                self.sel_a = self.graph.get_operation_by_name('ProdEnvMatR').get_attr('sel')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('ProdEnvMatR')
            except KeyError:
                self.sel_a = self.graph.get_operation_by_name('DescrptSeR').get_attr('sel')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('DescrptSeR')
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            try:
                self.sel_a = self.graph.get_operation_by_name('ProdEnvMatA').get_attr('sel_a')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('ProdEnvMatA')
            except KeyError:
                self.sel_a = self.graph.get_operation_by_name('DescrptSeA').get_attr('sel_a')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('DescrptSeA')
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            try:
                self.sel_a = self.graph.get_operation_by_name('ProdEnvMatA').get_attr('sel_a')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('ProdEnvMatA')
            except KeyError:
                self.sel_a = self.graph.get_operation_by_name('DescrptSeA').get_attr('sel_a')
                self.prod_env_mat_op = self.graph.get_operation_by_name ('DescrptSeA')
        else:
            raise RuntimeError("Unsupported descriptor")

        self.davg = get_tensor_by_name_from_graph(self.graph, f'descrpt_attr{self.suffix}/t_avg')
        self.dstd = get_tensor_by_name_from_graph(self.graph, f'descrpt_attr{self.suffix}/t_std')
        self.ntypes = get_tensor_by_name_from_graph(self.graph, 'descrpt_attr/ntypes')

        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            self.rcut = self.prod_env_mat_op.get_attr('rcut')
            self.rcut_smth = self.prod_env_mat_op.get_attr('rcut_smth')
        else:
            self.rcut = self.prod_env_mat_op.get_attr('rcut_r')
            self.rcut_smth = self.prod_env_mat_op.get_attr('rcut_r_smth')

        self.embedding_net_nodes = get_embedding_net_nodes_from_graph_def(self.graph_def, suffix=self.suffix)

        # move it to the descriptor class
        # for tt in self.exclude_types:
        #     if (tt[0] not in range(self.ntypes)) or (tt[1] not in range(self.ntypes)):
        #         raise RuntimeError("exclude types" + str(tt) + " must within the number of atomic types " + str(self.ntypes) + "!")
        # if (self.ntypes * self.ntypes - len(self.exclude_types) == 0):
        #     raise RuntimeError("empty embedding-net are not supported in model compression!")
        self.layer_size = self._get_layer_size()
        self.table_size = self._get_table_size()

        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        self.data_type  = self._get_data_type()
        self.last_layer_size = self._get_last_layer_size()

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
        neuron
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`

        Returns
        ----------
        lower
                The lower boundary of environment matrix
        upper
                The upper boundary of environment matrix
        """
        # tabulate range [lower, upper] with stride0 'stride0'
        lower, upper = self._get_env_mat_range(min_nbor_dist)

        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            xx = np.arange(lower, upper, stride0, dtype = self.data_type)
            xx = np.append(xx, np.arange(upper, extrapolate * upper, stride1, dtype = self.data_type))
            xx = np.append(xx, np.array([extrapolate * upper], dtype = self.data_type))
            self.nspline = int((upper - lower) / stride0 + (extrapolate * upper - upper) / stride1)
            for ii in range(self.table_size):
                if self.type_one_side or (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                    else:
                        net = "filter_" + str(ii // self.ntypes) + "_net_" + str(ii % self.ntypes)
                    self._build_lower(net, xx, ii, upper, lower, stride0, stride1, extrapolate)
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            xx = np.arange(extrapolate * lower, lower, stride1, dtype = self.data_type)
            xx = np.append(xx, np.arange(lower, upper, stride0, dtype = self.data_type))
            xx = np.append(xx, np.arange(upper, extrapolate * upper, stride1, dtype = self.data_type))
            xx = np.append(xx, np.array([extrapolate * upper], dtype = self.data_type))
            self.nspline = int((upper - lower) / stride0 + 2 * ((extrapolate * upper - upper) / stride1))
            idx = 0
            for ii in range(self.ntypes):
                for jj in range(ii, self.ntypes):
                    net = "filter_" + str(ii) + "_net_" + str(jj)
                    self._build_lower(net, xx, idx, upper, lower, stride0, stride1, extrapolate)
                    idx += 1
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            xx = np.arange(lower, upper, stride0, dtype = self.data_type)
            xx = np.append(xx, np.arange(upper, extrapolate * upper, stride1, dtype = self.data_type))
            xx = np.append(xx, np.array([extrapolate * upper], dtype = self.data_type))
            self.nspline = int((upper - lower) / stride0 + (extrapolate * upper - upper) / stride1)
            for ii in range(self.table_size):
                if self.type_one_side or (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                    else:
                        net = "filter_" + str(ii // self.ntypes) + "_net_" + str(ii % self.ntypes)
                    self._build_lower(net, xx, ii, upper, lower, stride0, stride1, extrapolate)
        else:
            raise RuntimeError("Unsupported descriptor")

        return lower, upper

    def _build_lower(self, net, xx, idx, upper, lower, stride0, stride1, extrapolate):
        vv, dd, d2 = self._make_data(xx, idx)
        self.data[net] = np.zeros([self.nspline, 6 * self.last_layer_size], dtype = self.data_type)

        # tt.shape: [self.nspline, self.last_layer_size]
        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            tt = np.full((self.nspline, self.last_layer_size), stride1)
            tt[:int((upper - lower) / stride0), :] = stride0
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            tt = np.full((self.nspline, self.last_layer_size), stride1)
            tt[int((lower - extrapolate * lower) / stride1) + 1:(int((lower - extrapolate * lower) / stride1) + int((upper - lower) / stride0)), :] = stride0
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            tt = np.full((self.nspline, self.last_layer_size), stride1)
            tt[:int((upper - lower) / stride0), :] = stride0
        else:
            raise RuntimeError("Unsupported descriptor")

        # hh.shape: [self.nspline, self.last_layer_size]
        hh = vv[1:self.nspline+1, :self.last_layer_size] - vv[:self.nspline, :self.last_layer_size]

        self.data[net][:, :6 * self.last_layer_size:6] = vv[:self.nspline, :self.last_layer_size]
        self.data[net][:, 1:6 * self.last_layer_size:6] = dd[:self.nspline, :self.last_layer_size]
        self.data[net][:, 2:6 * self.last_layer_size:6] = 0.5 * d2[:self.nspline, :self.last_layer_size]
        self.data[net][:, 3:6 * self.last_layer_size:6] = (1 / (2 * tt * tt * tt)) * (20 * hh - (8 * dd[1:self.nspline+1, :self.last_layer_size] + 12 * dd[:self.nspline, :self.last_layer_size]) * tt - (3 * d2[:self.nspline, :self.last_layer_size] - d2[1:self.nspline+1, :self.last_layer_size]) * tt * tt)
        self.data[net][:, 4:6 * self.last_layer_size:6] = (1 / (2 * tt * tt * tt * tt)) * (-30 * hh + (14 * dd[1:self.nspline+1, :self.last_layer_size] + 16 * dd[:self.nspline, :self.last_layer_size]) * tt + (3 * d2[:self.nspline, :self.last_layer_size] - 2 * d2[1:self.nspline+1, :self.last_layer_size]) * tt * tt)
        self.data[net][:, 5:6 * self.last_layer_size:6] = (1 / (2 * tt * tt * tt * tt * tt)) * (12 * hh - 6 * (dd[1:self.nspline+1, :self.last_layer_size] + dd[:self.nspline, :self.last_layer_size]) * tt + (d2[1:self.nspline+1, :self.last_layer_size] - d2[:self.nspline, :self.last_layer_size]) * tt * tt)

    def _load_sub_graph(self):
        sub_graph_def = tf.GraphDef()
        with tf.Graph().as_default() as sub_graph:
            tf.import_graph_def(sub_graph_def, name = "")
        return sub_graph, sub_graph_def

    def _get_bias(self):
        bias = {}
        for layer in range(1, self.layer_size + 1):
            bias["layer_" + str(layer)] = []
            if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/bias_{layer}_{ii}"]
                        bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                            node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/bias_{layer}_{ii % self.ntypes}"]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/bias_{layer}_{ii}_{jj}"]
                        bias["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/bias_{layer}_{ii}"]
                        bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                            node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/bias_{layer}_{ii % self.ntypes}"]
                            bias["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            bias["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")
        return bias

    def _get_matrix(self):
        matrix = {}
        for layer in range(1, self.layer_size + 1):
            matrix["layer_" + str(layer)] = []
            if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/matrix_{layer}_{ii}"]
                        matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                            node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/matrix_{layer}_{ii % self.ntypes}"]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
            elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
                for ii in range(self.ntypes):
                    for jj in range(ii, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/matrix_{layer}_{ii}_{jj}"]
                        matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
            elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
                if self.type_one_side:
                    for ii in range(0, self.ntypes):
                        node = self.embedding_net_nodes[f"filter_type_all{self.suffix}/matrix_{layer}_{ii}"]
                        matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                else:
                    for ii in range(0, self.ntypes * self.ntypes):
                        if (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types:
                            node = self.embedding_net_nodes[f"filter_type_{ii // self.ntypes}{self.suffix}/matrix_{layer}_{ii % self.ntypes}"]
                            matrix["layer_" + str(layer)].append(tf.make_ndarray(node))
                        else:
                            matrix["layer_" + str(layer)].append(np.array([]))
            else:
                raise RuntimeError("Unsupported descriptor")

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
                        if self.neuron[0] == 1:
                            yy = self._layer_0(
                                xx, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx]) + xx
                            dy = op_module.unaggregated_dy_dx_s(
                                yy, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype)) + tf.ones([1, 1], yy.dtype)
                            dy2 = op_module.unaggregated_dy2_dx_s(
                                yy, dy, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype))
                        elif self.neuron[0] == 2:
                            tt, yy = self._layer_1(
                                xx, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                            dy = op_module.unaggregated_dy_dx_s(
                                yy - tt, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype)) + tf.ones([1, 2], yy.dtype)
                            dy2 = op_module.unaggregated_dy2_dx_s(
                                yy - tt, dy, self.matrix["layer_" + str(layer + 1)][idx], xbar, tf.constant(self.functype))
                        else:
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

    # Change the embedding net range to sw / min_nbor_dist
    def _get_env_mat_range(self,
                           min_nbor_dist):
        lower = +100.0
        upper = -100.0
        sw    = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            lower = np.min(-self.davg[:, 0] / self.dstd[:, 0])
            upper = np.max(((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0])
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            var = np.square(sw / (min_nbor_dist * self.dstd[:, 1:4]))
            lower = np.min(-var)
            upper = np.max(var)
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            lower = np.min(-self.davg[:, 0] / self.dstd[:, 0])
            upper = np.max(((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0])
        else:
            raise RuntimeError("Unsupported descriptor")
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

    def _get_layer_size(self):
        layer_size = 0
        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            layer_size = len(self.embedding_net_nodes) // ((self.ntypes * self.ntypes - len(self.exclude_types)) * 2)
            if self.type_one_side :
                layer_size = len(self.embedding_net_nodes) // (self.ntypes * 2)
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            layer_size = len(self.embedding_net_nodes) // int(comb(self.ntypes + 1, 2) * 2)
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            layer_size = len(self.embedding_net_nodes) // ((self.ntypes * self.ntypes - len(self.exclude_types)) * 2)
            if self.type_one_side :
                layer_size = len(self.embedding_net_nodes) // (self.ntypes * 2)
        else:
            raise RuntimeError("Unsupported descriptor")
        return layer_size

    def _get_table_size(self):
        table_size = 0
        if isinstance(self.descrpt, deepmd.descriptor.DescrptSeA):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side :
                table_size = self.ntypes
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeT):
            table_size = int(comb(self.ntypes + 1, 2))
        elif isinstance(self.descrpt, deepmd.descriptor.DescrptSeR):
            table_size = self.ntypes * self.ntypes
            if self.type_one_side :
                table_size = self.ntypes
        else:
            raise RuntimeError("Unsupported descriptor")
        return table_size
    
    def _get_data_type(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return type(item[0][0])
        return None

    def _get_last_layer_size(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return item.shape[1]
        return 0