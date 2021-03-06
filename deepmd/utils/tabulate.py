import re
import math
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from deepmd.env import tf
from deepmd.env import op_module
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

log = logging.getLogger(__name__)

class DeepTabulate():
    """
    Class for tabulation.
    Compress a model, which including tabulating the embedding-net. 
    The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first table takes the stride(parameter) as it\'s uniform stride, while the second table takes 10 * stride as it\s uniform stride 
    The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table\'s upper boundary(upper) to the extrapolate(parameter) * upper.
    """
    def __init__(self,
                 model_file : str,
                 type_one_side : bool = False) -> None:
        """
        Constructor

        Parameters
        ----------
        model_file
                The frozen model
        type_one_side
                Try to build N_types tables. Otherwise, building N_types^2 tables
        """

        self.model_file = model_file
        self.type_one_side = type_one_side

        self.graph, self.graph_def = self._load_graph()
        self.sess = tf.Session(graph = self.graph)

        self.sub_graph, self.sub_graph_def = self._load_sub_graph()
        self.sub_sess = tf.Session(graph = self.sub_graph)

        self.sel_a = self.graph.get_operation_by_name('DescrptSeA').get_attr('sel_a')
        self.ntypes = self._get_tensor_value(self.graph.get_tensor_by_name ('descrpt_attr/ntypes:0'))

        self.davg = self._get_tensor_value(self.graph.get_tensor_by_name ('descrpt_attr/t_avg:0'))
        self.dstd = self._get_tensor_value(self.graph.get_tensor_by_name ('descrpt_attr/t_std:0'))

        self.descrpt = self.graph.get_operation_by_name ('DescrptSeA')
        self.rcut = self.descrpt.get_attr('rcut_r')
        self.rcut_smth = self.descrpt.get_attr('rcut_r_smth')

        self.filter_variable_nodes = self._load_matrix_node()
        self.layer_size = int(len(self.filter_variable_nodes) / (self.ntypes * self.ntypes * 2))
        self.table_size = self.ntypes * self.ntypes
        if type_one_side :
            self.layer_size = int(len(self.filter_variable_nodes) / (self.ntypes * 2))
            self.table_size = self.ntypes
        # self.value_type = self.filter_variable_nodes["filter_type_0/matrix_1_0"].dtype #"filter_type_0/matrix_1_0" must exit~
        # get trained variables
        self.bias = self._get_bias()
        self.matrix = self._get_matrix()

        self.data_type = type(self.matrix["layer_1"][0][0][0])
        assert self.matrix["layer_1"][0].size > 0, "no matrix exist in matrix array!"
        self.last_layer_size = self.matrix["layer_" + str(self.layer_size)][0].shape[1]
        # define tables
        self.data = {}

        # TODO: Need a check function to determine if the current model is properly

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
            vv, dd, d2 = self._make_data(xx, ii)
            if self.type_one_side:
                net = "filter_-1_net_" + str(int(ii))
            else:
                net = "filter_" + str(int(ii / self.ntypes)) + "_net_" + str(int(ii % self.ntypes))
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
            self.data[net]
        return lower, upper

    def _load_graph(self):
        graph_def = tf.GraphDef()
        with open(self.model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name = "")
        return graph, graph_def

    def _load_sub_graph(self):
        sub_graph_def = tf.GraphDef()
        with tf.Graph().as_default() as sub_graph:
            tf.import_graph_def(sub_graph_def, name = "")
        return sub_graph, sub_graph_def

    def _get_tensor_value(self, tensor) :
        with self.sess.as_default():
            self.sess.run(tensor)
            value = tensor.eval()
        return value

    def _load_matrix_node(self):
        matrix_node = {}
        matrix_node_pattern = "filter_type_\d+/matrix_\d+_\d+|filter_type_\d+/bias_\d+_\d+|filter_type_\d+/idt_\d+_\d+|filter_type_all/matrix_\d+_\d+|filter_type_all/bias_\d+_\d+|filter_type_all/idt_\d+_\d"
        for node in self.graph_def.node:
            if re.fullmatch(matrix_node_pattern, node.name) != None:
                matrix_node[node.name] = node.attr["value"].tensor
        for key in matrix_node.keys() :
            assert key.find('bias') > 0 or key.find('matrix') > 0, "currently, only support weight matrix and bias matrix at the tabulation op!"
        return matrix_node

    def _get_bias(self):
        bias = {}
        for layer in range(1, self.layer_size + 1):
            bias["layer_" + str(layer)] = []
            if self.type_one_side:
                for ii in range(0, self.ntypes):
                    tensor_value = np.frombuffer (self.filter_variable_nodes["filter_type_all/bias_" + str(layer) + "_" + str(int(ii))].tensor_content)
                    tensor_shape = tf.TensorShape(self.filter_variable_nodes["filter_type_all/bias_" + str(layer) + "_" + str(int(ii))].tensor_shape).as_list()
                    bias["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
            else:
                for ii in range(0, self.ntypes * self.ntypes):
                    tensor_value = np.frombuffer(self.filter_variable_nodes["filter_type_" + str(int(ii / self.ntypes)) + "/bias_" + str(layer) + "_" + str(int(ii % self.ntypes))].tensor_content)
                    tensor_shape = tf.TensorShape(self.filter_variable_nodes["filter_type_" + str(int(ii / self.ntypes)) + "/bias_" + str(layer) + "_" + str(int(ii % self.ntypes))].tensor_shape).as_list()
                    bias["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
        return bias

    def _get_matrix(self):
        matrix = {}
        for layer in range(1, self.layer_size + 1):
            matrix["layer_" + str(layer)] = []
            if self.type_one_side:
                for ii in range(0, self.ntypes):
                    tensor_value = np.frombuffer (self.filter_variable_nodes["filter_type_all/matrix_" + str(layer) + "_" + str(int(ii))].tensor_content)
                    tensor_shape = tf.TensorShape(self.filter_variable_nodes["filter_type_all/matrix_" + str(layer) + "_" + str(int(ii))].tensor_shape).as_list()
                    matrix["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
            else:
                for ii in range(0, self.ntypes * self.ntypes):
                    tensor_value = np.frombuffer(self.filter_variable_nodes["filter_type_" + str(int(ii / self.ntypes)) + "/matrix_" + str(layer) + "_" + str(int(ii % self.ntypes))].tensor_content)
                    tensor_shape = tf.TensorShape(self.filter_variable_nodes["filter_type_" + str(int(ii / self.ntypes)) + "/matrix_" + str(layer) + "_" + str(int(ii % self.ntypes))].tensor_shape).as_list()
                    matrix["layer_" + str(layer)].append(np.reshape(tensor_value, tensor_shape))
        return matrix

    # one-by-one executions
    def _make_data(self, xx, idx):
        with self.sub_graph.as_default():
            with self.sub_sess.as_default():
                xx = tf.reshape(xx, [xx.size, -1])
                for layer in range(self.layer_size):
                    if layer == 0:
                        yy = self._layer_0(xx, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                        dy = op_module.unaggregated_dy_dx_s(yy, self.matrix["layer_" + str(layer + 1)][idx])
                        dy2 = op_module.unaggregated_dy2_dx_s(yy, dy, self.matrix["layer_" + str(layer + 1)][idx])
                    else:
                        tt, yy = self._layer_1(yy, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                        dz = op_module.unaggregated_dy_dx(yy - tt, self.matrix["layer_" + str(layer + 1)][idx], dy)
                        dy2 = op_module.unaggregated_dy2_dx(yy - tt, self.matrix["layer_" + str(layer + 1)][idx], dz, dy, dy2)
                        dy = dz
 
                vv = yy.eval()
                dd = dy.eval()
                d2 = dy2.eval()
        return vv, dd, d2

    def _layer_0(self, x, w, b):
        return tf.nn.tanh(tf.matmul(x, w) + b)

    def _layer_1(self, x, w, b):
        t = tf.concat([x, x], axis = 1)
        return t, tf.nn.tanh(tf.matmul(x, w) + b) + t

    def _save_data(self):
        for ii in range(self.ntypes * self.ntypes):
            net = "filter_" + str(int(ii / self.ntypes)) + "_net_" + str(int(ii % self.ntypes))
            np.savetxt('data_' + str(int(ii)), self.data[net])

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