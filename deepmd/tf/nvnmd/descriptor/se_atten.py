# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
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
    ntype = nvnmd_cfg.dscp["ntype"]
    NIDP = nvnmd_cfg.dscp["NIDP"]
    ndescrpt = NIDP * 4
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
    if (davg is not None) or (dstd is not None):
        if davg is None:
            davg = np.zeros([ntype, ndescrpt])
        if dstd is None:
            dstd = np.ones([ntype, ndescrpt])
        nvnmd_cfg.get_s_range(davg, dstd)


def build_op_descriptor():
    r"""Replace se_a.py/DescrptSeA/build."""
    if nvnmd_cfg.quantize_descriptor:
        return op_module.prod_env_mat_a_mix_nvnmd_quantize
    else:
        return op_module.prod_env_mat_a_mix


def descrpt2r4(inputs, atype):
    r"""Replace :math:`r_{ji} \rightarrow r'_{ji}`
    where :math:`r_{ji} = (x_{ji}, y_{ji}, z_{ji})` and
    :math:`r'_{ji} = (s_{ji}, \frac{s_{ji} x_{ji}}{r_{ji}}, \frac{s_{ji} y_{ji}}{r_{ji}}, \frac{s_{ji} z_{ji}}{r_{ji}})`.
    """
    NIDP = nvnmd_cfg.dscp["NIDP"]
    ndescrpt = NIDP * 4

    # (nf*na*ni, 4)
    inputs_reshape = tf.reshape(inputs, [-1, 4])

    # u (i.e., r^2)
    u = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
    with tf.variable_scope("u", reuse=True):
        u = op_module.flt_nvnmd(u)
        log.debug("#u: %s", u)
        u = tf.ensure_shape(u, [None, 1])
    # rji
    rji = tf.reshape(tf.slice(inputs_reshape, [0, 1], [-1, 3]), [-1, 3])
    with tf.variable_scope("rji", reuse=True):
        rji = op_module.flt_nvnmd(rji)
        rji = tf.ensure_shape(rji, [None, 3])
        log.debug("#rji: %s", rji)

    # s & h
    u = tf.reshape(u, [-1, 1])
    table = GLOBAL_NP_FLOAT_PRECISION(
        np.concatenate([nvnmd_cfg.map["s"][0], nvnmd_cfg.map["h"][0]], axis=1)
    )
    table_grad = GLOBAL_NP_FLOAT_PRECISION(
        np.concatenate(
            [nvnmd_cfg.map["s_grad"][0], nvnmd_cfg.map["h_grad"][0]],
            axis=1,
        )
    )
    table_info = nvnmd_cfg.map["cfg_u2s"]
    table_info = np.array([np.float64(v) for vs in table_info for v in vs])
    table_info = GLOBAL_NP_FLOAT_PRECISION(table_info)

    s_h = op_module.map_flt_nvnmd(u, table, table_grad, table_info)
    s_h = tf.ensure_shape(s_h, [None, 1, 2])
    s = tf.slice(s_h, [0, 0, 0], [-1, -1, 1])
    h = tf.slice(s_h, [0, 0, 1], [-1, -1, 1])
    s = tf.reshape(s, [-1, 1])
    h = tf.reshape(h, [-1, 1])

    with tf.variable_scope("s_s", reuse=True):
        s = op_module.flt_nvnmd(s)
        log.debug("#s_s: %s", s)
        s = tf.ensure_shape(s, [None, 1])

    with tf.variable_scope("h_s", reuse=True):
        h = op_module.flt_nvnmd(h)
        log.debug("#h_s: %s", h)
        h = tf.ensure_shape(h, [None, 1])
    # davg and dstd
    # davg = nvnmd_cfg.map["davg"]  # is_zero
    dstd_inv = nvnmd_cfg.map["dstd_inv"]
    atype_expand = tf.reshape(atype, [-1, 1])
    std_inv_sel = tf.nn.embedding_lookup(dstd_inv, atype_expand)
    std_inv_sel = tf.reshape(std_inv_sel, [-1, 4])
    std_inv_s = tf.slice(std_inv_sel, [0, 0], [-1, 1])
    std_inv_h = tf.slice(std_inv_sel, [0, 1], [-1, 1])
    s = op_module.mul_flt_nvnmd(std_inv_s, tf.reshape(s, [-1, NIDP]))
    h = op_module.mul_flt_nvnmd(std_inv_h, tf.reshape(h, [-1, NIDP]))
    s = tf.ensure_shape(s, [None, NIDP])
    h = tf.ensure_shape(h, [None, NIDP])
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
    # Rxyz = h * rji
    Rxyz = op_module.mul_flt_nvnmd(h, rji)
    Rxyz = tf.ensure_shape(Rxyz, [None, 3])
    with tf.variable_scope("Rxyz", reuse=True):
        Rxyz = op_module.flt_nvnmd(Rxyz)
        log.debug("#Rxyz: %s", Rxyz)
        Rxyz = tf.ensure_shape(Rxyz, [None, 3])
    R4 = tf.concat([Rs, Rxyz], axis=1)
    inputs_reshape = R4
    inputs_reshape = tf.reshape(inputs_reshape, [-1, ndescrpt])
    return inputs_reshape


def filter_lower_R42GR(inputs_i, atype, nei_type_vec):
    r"""Replace se_a.py/DescrptSeA/_filter_lower."""
    shape_i = inputs_i.get_shape().as_list()
    inputs_reshape = tf.reshape(inputs_i, [-1, 4])
    M1 = nvnmd_cfg.dscp["M1"]
    ntype = nvnmd_cfg.dscp["ntype"]
    NIDP = nvnmd_cfg.dscp["NIDP"]
    two_embd_value = nvnmd_cfg.map["gt"]
    # print(two_embd_value)

    # copy
    inputs_reshape = op_module.flt_nvnmd(inputs_reshape)
    inputs_reshape = tf.ensure_shape(inputs_reshape, [None, 4])

    inputs_reshape, inputs_reshape2 = op_module.copy_flt_nvnmd(inputs_reshape)
    inputs_reshape = tf.ensure_shape(inputs_reshape, [None, 4])
    inputs_reshape2 = tf.ensure_shape(inputs_reshape2, [None, 4])
    # s2G
    s = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
    table = GLOBAL_NP_FLOAT_PRECISION(nvnmd_cfg.map["g"][0])
    table_grad = GLOBAL_NP_FLOAT_PRECISION(nvnmd_cfg.map["g_grad"][0])
    table_info = nvnmd_cfg.map["cfg_s2g"]
    table_info = np.array([np.float64(v) for vs in table_info for v in vs])
    table_info = GLOBAL_NP_FLOAT_PRECISION(table_info)
    G = op_module.map_flt_nvnmd(s, table, table_grad, table_info)
    G = tf.ensure_shape(G, [None, 1, M1])
    with tf.variable_scope("g_s", reuse=True):
        G = op_module.flt_nvnmd(G)
        log.debug("#g_s: %s", G)
        G = tf.ensure_shape(G, [None, 1, M1])
    # t2G
    atype_expand = tf.reshape(atype, [-1, 1])
    idx_i = tf.tile(atype_expand * (ntype + 1), [1, NIDP])
    idx_j = tf.reshape(nei_type_vec, [-1, NIDP])
    idx = idx_i + idx_j
    index_of_two_side = tf.reshape(idx, [-1])
    two_embd = tf.nn.embedding_lookup(two_embd_value, index_of_two_side)
    # two_embd = tf.reshape(two_embd, (-1, shape_i[1] // 4, M1))
    two_embd = tf.reshape(two_embd, (-1, M1))
    with tf.variable_scope("g_t", reuse=True):
        two_embd = op_module.flt_nvnmd(two_embd)
        log.debug("#g_t: %s", two_embd)
        two_embd = tf.ensure_shape(two_embd, [None, M1])
    # G_s, G_t -> G
    G = tf.reshape(G, [-1, M1])
    G = op_module.mul_flt_nvnmd(G, two_embd)
    G = tf.ensure_shape(G, [None, M1])
    with tf.variable_scope("g", reuse=True):
        G = op_module.flt_nvnmd(G)
        log.debug("#g: %s", G)
        G = tf.ensure_shape(G, [None, M1])
    xyz_scatter = G
    xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, M1))
    # GR
    inputs_reshape2 = tf.reshape(inputs_reshape2, [-1, shape_i[1] // 4, 4])
    GR = op_module.matmul_flt2fix_nvnmd(
        tf.transpose(inputs_reshape2, [0, 2, 1]), xyz_scatter, 23
    )
    GR = tf.ensure_shape(GR, [None, 4, M1])
    return GR


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
