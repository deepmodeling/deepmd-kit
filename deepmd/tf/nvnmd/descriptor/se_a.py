# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)

#
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.nvnmd.utils.weight import (
    get_normalize,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.network import (
    embedding_net,
)

log = logging.getLogger(__name__)


def build_davg_dstd():
    r"""Get the davg and dstd from the dictionary nvnmd_cfg.
    The davg and dstd have been obtained by training CNN.
    """
    davg, dstd = get_normalize(nvnmd_cfg.weight)
    return davg, dstd


def check_switch_range(davg, dstd):
    r"""Check the range of switch, let it in range [-2, 14]."""
    rmin = nvnmd_cfg.dscp["rcut_smth"]
    #
    namelist = [n.name for n in tf.get_default_graph().as_graph_def().node]
    if "train_attr/min_nbor_dist" in namelist:
        min_dist = get_tensor_by_name_from_graph(
            tf.get_default_graph(), "train_attr/min_nbor_dist"
        )
    elif "train_attr.min_nbor_dist" in nvnmd_cfg.weight.keys():
        if nvnmd_cfg.weight["train_attr.min_nbor_dist"] < 1e-6:
            min_dist = rmin
        else:
            min_dist = nvnmd_cfg.weight["train_attr.min_nbor_dist"]
    else:
        min_dist = None

    # fix the bug: if model initial mode is 'init_from_model',
    # we need dmin to calculate smin and smax in mapt.py
    if min_dist is not None:
        nvnmd_cfg.dscp["dmin"] = min_dist
        nvnmd_cfg.save()

    # if davg and dstd is None, the model initial mode is in
    #  'init_from_model', 'restart', 'init_from_frz_model', 'finetune'
    if (davg is not None) and (dstd is not None):
        nvnmd_cfg.get_s_range(davg, dstd)


def build_op_descriptor():
    r"""Replace se_a.py/DescrptSeA/build."""
    if nvnmd_cfg.quantize_descriptor:
        return op_module.prod_env_mat_a_nvnmd_quantize
    else:
        return op_module.prod_env_mat_a


def descrpt2r4(inputs, natoms):
    r"""Replace :math:`r_{ji} \rightarrow r'_{ji}`
    where :math:`r_{ji} = (x_{ji}, y_{ji}, z_{ji})` and
    :math:`r'_{ji} = (s_{ji}, \frac{s_{ji} x_{ji}}{r_{ji}}, \frac{s_{ji} y_{ji}}{r_{ji}}, \frac{s_{ji} z_{ji}}{r_{ji}})`.
    """
    ntypes = nvnmd_cfg.dscp["ntype"]
    NIDP = nvnmd_cfg.dscp["NIDP"]
    ndescrpt = NIDP * 4
    start_index = 0

    # (nf*na*ni, 4)
    inputs_reshape = tf.reshape(inputs, [-1, 4])

    with tf.variable_scope("filter_type_all_x", reuse=True):
        # u (i.e., r^2)
        u = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        with tf.variable_scope("u", reuse=True):
            u = op_module.flt_nvnmd(u)
            log.debug("#u: %s", u)
            u = tf.ensure_shape(u, [None, 1])
        u = tf.reshape(u, [-1, natoms[0] * NIDP])
        sh0 = tf.shape(u)[0]
        # rij
        rij = tf.reshape(tf.slice(inputs_reshape, [0, 1], [-1, 3]), [-1, 3])
        with tf.variable_scope("rij", reuse=True):
            rij = op_module.flt_nvnmd(rij)
            rij = tf.ensure_shape(rij, [None, 3])
            log.debug("#rij: %s", rij)
        s = []
        h = []
        for type_i in range(ntypes):
            type_input = 0
            u_i = tf.slice(u, [0, start_index * NIDP], [-1, natoms[2 + type_i] * NIDP])
            u_i = tf.reshape(u_i, [-1, 1])
            # s
            table = GLOBAL_NP_FLOAT_PRECISION(
                np.concatenate(
                    [nvnmd_cfg.map["s"][type_i], nvnmd_cfg.map["h"][type_i]], axis=1
                )
            )
            table_grad = GLOBAL_NP_FLOAT_PRECISION(
                np.concatenate(
                    [nvnmd_cfg.map["s_grad"][type_i], nvnmd_cfg.map["h_grad"][type_i]],
                    axis=1,
                )
            )
            table_info = nvnmd_cfg.map["cfg_u2s"]
            table_info = np.array([np.float64(v) for vs in table_info for v in vs])
            table_info = GLOBAL_NP_FLOAT_PRECISION(table_info)

            s_h_i = op_module.map_flt_nvnmd(u_i, table, table_grad, table_info)
            s_h_i = tf.ensure_shape(s_h_i, [None, 1, 2])
            s_i = tf.slice(s_h_i, [0, 0, 0], [-1, -1, 1])
            h_i = tf.slice(s_h_i, [0, 0, 1], [-1, -1, 1])
            # reshape shape to sh0 for fixing bug.
            # This bug occurs if the number of atoms of an element is zero.
            s_i = tf.reshape(s_i, [sh0, natoms[2 + type_i] * NIDP])
            h_i = tf.reshape(h_i, [sh0, natoms[2 + type_i] * NIDP])
            s.append(s_i)
            h.append(h_i)
            start_index += natoms[2 + type_i]

        s = tf.concat(s, axis=1)
        h = tf.concat(h, axis=1)

        s = tf.reshape(s, [-1, 1])
        h = tf.reshape(h, [-1, 1])

        with tf.variable_scope("s", reuse=True):
            s = op_module.flt_nvnmd(s)
            log.debug("#s: %s", s)
            s = tf.ensure_shape(s, [None, 1])

        with tf.variable_scope("h", reuse=True):
            h = op_module.flt_nvnmd(h)
            log.debug("#h: %s", h)
            h = tf.ensure_shape(h, [None, 1])

        # R2R4
        Rs = s
        # Rxyz = h * rij
        Rxyz = op_module.mul_flt_nvnmd(h, rij)
        Rxyz = tf.ensure_shape(Rxyz, [None, 3])
        with tf.variable_scope("Rxyz", reuse=True):
            Rxyz = op_module.flt_nvnmd(Rxyz)
            log.debug("#Rxyz: %s", Rxyz)
            Rxyz = tf.ensure_shape(Rxyz, [None, 3])
        R4 = tf.concat([Rs, Rxyz], axis=1)
        R4 = tf.reshape(R4, [-1, NIDP, 4])
        inputs_reshape = R4
        inputs_reshape = tf.reshape(inputs_reshape, [-1, ndescrpt])
    return inputs_reshape


def filter_lower_R42GR(
    type_i,
    type_input,
    inputs_i,
    is_exclude,
    activation_fn,
    bavg,
    stddev,
    trainable,
    suffix,
    seed,
    seed_shift,
    uniform_seed,
    filter_neuron,
    filter_precision,
    filter_resnet_dt,
    embedding_net_variables,
):
    r"""Replace se_a.py/DescrptSeA/_filter_lower."""
    shape_i = inputs_i.get_shape().as_list()
    inputs_reshape = tf.reshape(inputs_i, [-1, 4])
    natom = tf.shape(inputs_i)[0]
    M1 = nvnmd_cfg.dscp["M1"]

    type_input = 0 if (type_input < 0) else type_input

    if nvnmd_cfg.quantize_descriptor:
        # copy
        inputs_reshape = op_module.flt_nvnmd(inputs_reshape)
        inputs_reshape = tf.ensure_shape(inputs_reshape, [None, 4])

        inputs_reshape, inputs_reshape2 = op_module.copy_flt_nvnmd(inputs_reshape)
        inputs_reshape = tf.ensure_shape(inputs_reshape, [None, 4])
        inputs_reshape2 = tf.ensure_shape(inputs_reshape2, [None, 4])
        # s
        s = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        # G
        table = GLOBAL_NP_FLOAT_PRECISION(nvnmd_cfg.map["g"][type_i])
        table_grad = GLOBAL_NP_FLOAT_PRECISION(nvnmd_cfg.map["g_grad"][type_i])
        table_info = nvnmd_cfg.map["cfg_s2g"]
        table_info = np.array([np.float64(v) for vs in table_info for v in vs])
        table_info = GLOBAL_NP_FLOAT_PRECISION(table_info)
        with tf.variable_scope("g", reuse=True):
            G = op_module.map_flt_nvnmd(s, table, table_grad, table_info)
            G = tf.ensure_shape(G, [None, 1, M1])
            G = op_module.flt_nvnmd(G)
            G = tf.ensure_shape(G, [None, 1, M1])
            log.debug("#g: %s", G)
        # G
        xyz_scatter = G
        xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, M1))
        # GR
        inputs_reshape2 = tf.reshape(inputs_reshape2, [-1, shape_i[1] // 4, 4])
        GR = op_module.matmul_flt2fix_nvnmd(
            tf.transpose(inputs_reshape2, [0, 2, 1]), xyz_scatter, 23
        )
        GR = tf.ensure_shape(GR, [None, 4, M1])
        return GR

    else:
        xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        if nvnmd_cfg.restore_descriptor:
            trainable = False
            embedding_net_variables = {}
            for key in nvnmd_cfg.weight.keys():
                if "filter_type" in key:
                    key2 = key.replace(".", "/")
                    embedding_net_variables[key2] = nvnmd_cfg.weight[key]

        if not is_exclude:
            xyz_scatter = embedding_net(
                xyz_scatter,
                filter_neuron,
                filter_precision,
                activation_fn=activation_fn,
                resnet_dt=filter_resnet_dt,
                name_suffix=suffix,
                stddev=stddev,
                bavg=bavg,
                seed=seed,
                trainable=trainable,
                uniform_seed=uniform_seed,
                initial_variables=embedding_net_variables,
            )
            if (not uniform_seed) and (seed is not None):
                seed += seed_shift
        else:
            # we can safely return the final xyz_scatter filled with zero directly
            return tf.cast(tf.fill((natom, 4, M1), 0.0), GLOBAL_TF_FLOAT_PRECISION)
        # natom x nei_type_i x out_size
        xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, M1))
        # When using tf.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
        # [588 24] -> [588 6 4] correct
        # but if sel is zero
        # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
        # So we need to explicitly assign the shape to tf.shape(inputs_i)[0] instead of -1
        return tf.matmul(
            tf.reshape(inputs_i, [natom, shape_i[1] // 4, 4]),
            xyz_scatter,
            transpose_a=True,
        )


def filter_GR2D(xyz_scatter_1):
    r"""Replace se_a.py/_filter."""
    NIX = nvnmd_cfg.dscp["NIX"]
    M1 = nvnmd_cfg.dscp["M1"]
    M2 = nvnmd_cfg.dscp["M2"]
    NBIT_DATA_FL = nvnmd_cfg.nbit["NBIT_FIXD_FL"]

    if nvnmd_cfg.quantize_descriptor:
        xyz_scatter_1 = tf.reshape(xyz_scatter_1, [-1, 4 * M1])
        # fix the number of bits of gradient
        xyz_scatter_1 = xyz_scatter_1 * (1.0 / NIX)

        with tf.variable_scope("gr", reuse=True):
            xyz_scatter_1 = op_module.flt_nvnmd(xyz_scatter_1)
            log.debug("#gr: %s", xyz_scatter_1)
            xyz_scatter_1 = tf.ensure_shape(xyz_scatter_1, [None, 4 * M1])
        xyz_scatter_1 = tf.reshape(xyz_scatter_1, [-1, 4, M1])

        # natom x 4 x outputs_size_2
        xyz_scatter_2 = xyz_scatter_1
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0, 1, 0], [-1, 3, -1])
        # natom x outputs_size_2 x 3
        qmat = tf.transpose(qmat, perm=[0, 2, 1])
        # D': natom x outputs_size x outputs_size_2
        xyz_scatter_1_T = tf.transpose(xyz_scatter_1, [0, 2, 1])
        result = op_module.matmul_flt_nvnmd(
            xyz_scatter_1_T, xyz_scatter_2, 1 * 16 + 0, 1 * 16 + 0
        )
        result = tf.ensure_shape(result, [None, M1, M1])
        # D': natom x (outputs_size x outputs_size_2)
        result = tf.reshape(result, [-1, M1 * M1])
        #
        index_subset = []
        for ii in range(M1):
            for jj in range(ii, ii + M2):
                index_subset.append((ii * M1) + (jj % M1))
        index_subset = tf.constant(np.int32(np.array(index_subset)))
        result = tf.gather(result, index_subset, axis=1)

        with tf.variable_scope("d", reuse=True):
            result = op_module.flt_nvnmd(result)
            log.debug("#d: %s", result)
            result = tf.ensure_shape(result, [None, M1 * M2])

        result = op_module.quantize_nvnmd(result, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
        result = tf.ensure_shape(result, [None, M1 * M2])
    else:
        # natom x 4 x outputs_size
        xyz_scatter_1 = xyz_scatter_1 * (1.0 / NIX)
        # natom x 4 x outputs_size_2
        # xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
        xyz_scatter_2 = xyz_scatter_1
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0, 1, 0], [-1, 3, -1])
        # natom x outputs_size_1 x 3
        qmat = tf.transpose(qmat, perm=[0, 2, 1])
        # natom x outputs_size x outputs_size_2
        result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a=True)
        # natom x (outputs_size x outputs_size_2)
        # result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])
        result = tf.reshape(result, [-1, M1 * M1])
        #
        index_subset = []
        for ii in range(M1):
            for jj in range(ii, ii + M2):
                index_subset.append((ii * M1) + (jj % M1))
        index_subset = tf.constant(np.int32(np.array(index_subset)))
        result = tf.gather(result, index_subset, axis=1)

    return result, qmat
