import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.utils.network import embedding_net


#
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.network import matmul3_qq
from deepmd.nvnmd.utils.weight import get_normalize, get_rng_s


def build_davg_dstd():
    r"""Get the davg and dstd from the dictionary nvnmd_cfg.
    The davg and dstd have been obtained by training CNN
    """
    davg, dstd = get_normalize(nvnmd_cfg.weight)
    return davg, dstd


def build_op_descriptor():
    r"""Replace se_a.py/DescrptSeA/build
    """
    if nvnmd_cfg.quantize_descriptor:
        return op_module.prod_env_mat_a_nvnmd_quantize
    else:
        return op_module.prod_env_mat_a


def descrpt2r4(inputs, natoms):
    r"""Replace :math:`r_{ji} \rightarrow r'_{ji}`
    where :math:`r_{ji} = (x_{ji}, y_{ji}, z_{ji})` and
    :math:`r'_{ji} = (s_{ji}, \frac{s_{ji} x_{ji}}{r_{ji}}, \frac{s_{ji} y_{ji}}{r_{ji}}, \frac{s_{ji} z_{ji}}{r_{ji}})`
    """
    NBIT_DATA_FL = nvnmd_cfg.nbit['NBIT_DATA_FL']
    NBIT_FEA_X_FL = nvnmd_cfg.nbit['NBIT_FEA_X_FL']
    NBIT_FEA_FL = nvnmd_cfg.nbit['NBIT_FEA_FL']
    prec = 1.0 / (2 ** NBIT_FEA_X_FL)

    ntypes = nvnmd_cfg.dscp['ntype']
    NIDP = nvnmd_cfg.dscp['NIDP']
    ndescrpt = NIDP * 4
    start_index = 0

    # (nf, na*nd)
    shape = inputs.get_shape().as_list()
    # (nf*na*ni, 4)
    inputs_reshape = tf.reshape(inputs, [-1, 4])

    with tf.variable_scope('filter_type_all_x', reuse=True):
        # u (i.e., r^2)
        u = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        with tf.variable_scope('u', reuse=True):
            u = op_module.quantize_nvnmd(u, 0, -1, NBIT_DATA_FL, -1)
        # print('u:', u)
        u = tf.reshape(u, [-1, natoms[0] * NIDP])
        # rij
        rij = tf.reshape(tf.slice(inputs_reshape, [0, 1], [-1, 3]), [-1, 3])
        with tf.variable_scope('rij', reuse=True):
            rij = op_module.quantize_nvnmd(rij, 0, NBIT_DATA_FL, -1, -1)
        # print('rij:', rij)
        s = []
        sr = []
        for type_i in range(ntypes):
            type_input = 0
            postfix = f"_t{type_input}_t{type_i}"
            u_i = tf.slice(
                u,
                [0, start_index * NIDP],
                [-1, natoms[2 + type_i] * NIDP])
            u_i = tf.reshape(u_i, [-1, 1])
            #
            keys = 's,sr'.split(',')
            map_tables = [nvnmd_cfg.map[key + postfix] for key in keys]
            map_tables2 = [nvnmd_cfg.map[f"d{key}_dr2" + postfix] for key in keys]
            map_outs = []
            for ii in range(len(keys)):
                map_outs.append(op_module.map_nvnmd(
                    u_i,
                    map_tables[ii][0],
                    map_tables[ii][1] / prec,
                    map_tables2[ii][0],
                    map_tables2[ii][1] / prec,
                    prec, NBIT_FEA_FL))

            s_i, sr_i = map_outs
            s_i = tf.reshape(s_i, [-1, natoms[2 + type_i] * NIDP])
            sr_i = tf.reshape(sr_i, [-1, natoms[2 + type_i] * NIDP])
            s.append(s_i)
            sr.append(sr_i)
            start_index += natoms[2 + type_i]

        s = tf.concat(s, axis=1)
        sr = tf.concat(sr, axis=1)

        with tf.variable_scope('s', reuse=True):
            s = op_module.quantize_nvnmd(s, 0, NBIT_FEA_FL, NBIT_DATA_FL, -1)

        with tf.variable_scope('sr', reuse=True):
            sr = op_module.quantize_nvnmd(sr, 0, NBIT_FEA_FL, NBIT_DATA_FL, -1)

        s = tf.reshape(s, [-1, 1])
        sr = tf.reshape(sr, [-1, 1])

        # R2R4
        Rs = s
        Rxyz = sr * rij
        with tf.variable_scope('Rxyz', reuse=True):
            Rxyz = op_module.quantize_nvnmd(Rxyz, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
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
        embedding_net_variables):
    r"""Replace se_a.py/DescrptSeA/_filter_lower
    """
    shape_i = inputs_i.get_shape().as_list()
    inputs_reshape = tf.reshape(inputs_i, [-1, 4])
    natom = tf.shape(inputs_i)[0]
    M1 = nvnmd_cfg.dscp['M1']

    NBIT_DATA_FL = nvnmd_cfg.nbit['NBIT_DATA_FL']
    NBIT_FEA_X_FL = nvnmd_cfg.nbit['NBIT_FEA_X_FL']
    NBIT_FEA_X2_FL = nvnmd_cfg.nbit['NBIT_FEA_X2_FL']
    NBIT_FEA_FL = nvnmd_cfg.nbit['NBIT_FEA_FL']
    prec = 1.0 / (2 ** NBIT_FEA_X2_FL)
    type_input = 0 if (type_input < 0) else type_input
    postfix = f"_t{type_input}_t{type_i}"

    if (nvnmd_cfg.quantize_descriptor):
        s_min, smax = get_rng_s(nvnmd_cfg.weight)
        s_min = -2.0
        # s_min = np.floor(s_min)
        s = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        s = op_module.quantize_nvnmd(s, 0, NBIT_FEA_FL, NBIT_DATA_FL, -1)
        # G
        keys = 'G'.split(',')
        map_tables = [nvnmd_cfg.map[key + postfix] for key in keys]
        map_tables2 = [nvnmd_cfg.map[f"d{key}_ds" + postfix] for key in keys]
        map_outs = []
        for ii in range(len(keys)):
            with tf.variable_scope(keys[ii], reuse=True):
                map_outs.append(op_module.map_nvnmd(
                    s - s_min,
                    map_tables[ii][0], map_tables[ii][1] / prec,
                    map_tables2[ii][0], map_tables2[ii][1] / prec,
                    prec, NBIT_FEA_FL))
                map_outs[ii] = op_module.quantize_nvnmd(map_outs[ii], 0, NBIT_FEA_FL, NBIT_DATA_FL, -1)
        G = map_outs
        # G
        xyz_scatter = G
        xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, M1))
        # GR
        inputs_reshape = tf.reshape(inputs_reshape, [-1, shape_i[1] // 4, 4])
        GR = matmul3_qq(tf.transpose(inputs_reshape, [0, 2, 1]), xyz_scatter, -1)
        GR = tf.reshape(GR, [-1, 4 * M1])
        return GR

    else:
        xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0, 0], [-1, 1]), [-1, 1])
        if nvnmd_cfg.restore_descriptor:
            trainable = False
            embedding_net_variables = {}
            for key in nvnmd_cfg.weight.keys():
                if 'filter_type' in key:
                    key2 = key.replace('.', '/')
                    embedding_net_variables[key2] = nvnmd_cfg.weight[key]

        if (not is_exclude):
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
                initial_variables=embedding_net_variables)
            if (not uniform_seed) and (seed is not None):
                seed += seed_shift
        else:
            # we can safely return the final xyz_scatter filled with zero directly
            return tf.cast(tf.fill((natom, 4, M1), 0.), GLOBAL_TF_FLOAT_PRECISION)
        # natom x nei_type_i x out_size
        xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1] // 4, M1))
        # When using tf.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
        # [588 24] -> [588 6 4] correct
        # but if sel is zero
        # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
        # So we need to explicitly assign the shape to tf.shape(inputs_i)[0] instead of -1
        return tf.matmul(tf.reshape(inputs_i, [natom, shape_i[1] // 4, 4]), xyz_scatter, transpose_a=True)


def filter_GR2D(xyz_scatter_1):
    r"""Replace se_a.py/_filter
    """
    NIX = nvnmd_cfg.dscp['NIX']
    NBIT_DATA_FL = nvnmd_cfg.nbit['NBIT_DATA_FL']
    M1 = nvnmd_cfg.dscp['M1']
    M2 = nvnmd_cfg.dscp['M2']

    if (nvnmd_cfg.quantize_descriptor):
        xyz_scatter_1 = tf.reshape(xyz_scatter_1, [-1, 4 * M1])
        # fix the number of bits of gradient
        xyz_scatter_1 = op_module.quantize_nvnmd(xyz_scatter_1, 0, -1, NBIT_DATA_FL, -1)
        xyz_scatter_1 = xyz_scatter_1 * (1.0 / NIX)
        with tf.variable_scope('GR', reuse=True):
            xyz_scatter_1 = op_module.quantize_nvnmd(xyz_scatter_1, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
        xyz_scatter_1 = tf.reshape(xyz_scatter_1, [-1, 4, M1])

        # natom x 4 x outputs_size_2
        xyz_scatter_2 = xyz_scatter_1
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0, 1, 0], [-1, 3, -1])
        # natom x outputs_size_2 x 3
        qmat = tf.transpose(qmat, perm=[0, 2, 1])
        # D': natom x outputs_size x outputs_size_2
        result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a=True)
        # D': natom x (outputs_size x outputs_size_2)
        result = tf.reshape(result, [-1, M1 * M1])
        #
        index_subset = []
        for ii in range(M1):
            for jj in range(ii, ii + M2):
                index_subset.append((ii * M1) + (jj % M1))
        index_subset = tf.constant(np.int32(np.array(index_subset)))
        result = tf.gather(result, index_subset, axis=1)

        with tf.variable_scope('d', reuse=True):
            result = op_module.quantize_nvnmd(result, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
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
