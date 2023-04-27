import logging
from typing import (
    Optional,
)

import numpy as np

from deepmd.env import (
    op_module,
    tf,
)
from deepmd.nvnmd.data.data import (
    jdata_deepmd_input,
    jdata_sys,
)
from deepmd.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.nvnmd.utils.fio import (
    FioDic,
)
from deepmd.nvnmd.utils.network import (
    get_sess,
)
from deepmd.nvnmd.utils.weight import (
    get_filter_weight,
    get_normalize,
)
from deepmd.utils.sess import (
    run_sess,
)

log = logging.getLogger(__name__)


class MapTable:
    r"""Generate the mapping table describing the relastionship of
    atomic distance, cutoff function, and embedding matrix.

    three mapping table will be built:

    | :math:`r^2_{ji} \rightarrow s_{ji}`
    | :math:`r^2_{ji} \rightarrow h_{ji}`
    | :math:`r^2_{ji} \rightarrow \mathcal{G}_{ji}`

    where :math:`s_{ji}` is cut-off function,
    :math:`h_{ji} = \frac{s(r_{ji})}{r_{ji}}`, and
    :math:`\mathcal{G}_{ji}` is embedding matrix.

    The mapping funciton can be define as:

    | :math:`y = f(x) = y_{k} + (x - x_{k}) * dy_{k}`
    | :math:`y_{k} = f(x_{k})`
    | :math:`dy_{k} = \frac{f(x_{k+1}) - f(x_{k})}{dx}`
    | :math:`x_{k} \leq x < x_{k+1}`
    | :math:`x_{k} = k * dx`

    where :math:`dx` is interpolation interval.

    Parameters
    ----------
    config_file
        input file name
        an .npy file containing the configuration information of NVNMD model
    weight_file
        input file name
        an .npy file containing the weights of NVNMD model
    map_file
        output file name
        an .npy file containing the mapping tables of NVNMD model

    References
    ----------
    DOI: 10.1038/s41524-022-00773-z
    """

    def __init__(self, config_file: str, weight_file: str, map_file: str):
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file

        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = config_file
        jdata["weight_file"] = weight_file
        jdata["enable"] = True

        nvnmd_cfg.init_from_jdata(jdata)

    def build_map(self):
        ntypex = nvnmd_cfg.dscp["ntypex"]
        ntype = nvnmd_cfg.dscp["ntype"]
        # calculate grid point
        dic_u2s, dic_u2s_ref = self.run_u2s()
        dic_s2g, dic_s2g_ref = self.run_s2g()
        # build mapping table
        rank = 4

        dic_map = {}
        u = dic_u2s["u"]
        cfg_u2s = [[u[0], u[512], u[1] - u[0], 0, 512]]
        dic_map["s"], dic_map["s_grad"] = self.build_map_coef(
            cfg_u2s,
            u,
            dic_u2s["s"],
            dic_u2s["s_grad"],
            dic_u2s["s_grad_grad"],
            ntype,
            1,
            rank,
        )
        dic_map["h"], dic_map["h_grad"] = self.build_map_coef(
            cfg_u2s,
            u,
            dic_u2s["h"],
            dic_u2s["h_grad"],
            dic_u2s["h_grad_grad"],
            ntype,
            1,
            rank,
        )

        dic_map2 = {}
        s = dic_s2g["s"]
        cfg_s2g = [
            [s[0], s[256], s[1] - s[0], 0, 256],
            [s[0], s[4096], s[16] - s[0], 256, 512],
        ]
        dic_map2["g"], dic_map2["g_grad"] = self.build_map_coef(
            cfg_s2g,
            s,
            dic_s2g["g"],
            dic_s2g["g_grad"],
            dic_s2g["g_grad_grad"],
            ntype,
            32,
            rank,
        )
        # run mapping to test
        if jdata_sys["debug"]:
            dic_u2s_prd = self.mapping2(dic_u2s_ref["u"], dic_map, cfg_u2s, rank)
            dic_s2g_prd = self.mapping2(dic_s2g_ref["s"], dic_map2, cfg_s2g, rank)

            self.plot_lines(dic_u2s_ref["u"], dic_u2s_prd, dic_u2s_ref)
            self.plot_lines(dic_s2g_ref["s"], dic_s2g_prd, dic_s2g_ref)
        # save
        self.map = {}
        self.map["cfg_u2s"] = cfg_u2s
        self.map["cfg_s2g"] = cfg_s2g
        self.map.update(dic_map)
        self.map.update(dic_map2)

        FioDic().save(self.map_file, self.map)
        log.info("NVNMD: finish building mapping table")
        return self.map

    def mapping(self, x, dic_map, cfgs, rank=4):
        r"""Evaluate value by mapping table operation of tensorflow."""
        n = len(x)
        dic_val = {}
        for key in dic_map.keys():
            val = dic_map[key]
            if isinstance(val, list):
                dats = []
                for ii in range(len(val)):
                    val_i = val[ii]
                    nr = np.shape(val_i)[0]
                    nc = np.shape(val_i)[1] // rank
                    dat_i = np.zeros([n, nc])
                    for kk in range(n):
                        xk = x[kk]
                        for cfg in cfgs:
                            x0, x1, dx, N0, N1 = cfg
                            if (xk >= x0) and (xk <= x1):
                                break
                        idx = np.int32(np.floor((xk - x0) / dx))
                        dxx = xk - idx * dx - x0
                        idx_k = idx + N0
                        dxx_k = dxx
                        if idx_k >= N1:
                            idx_k = N1 - 1
                            dxx_k = dx
                        coef = val_i[idx_k]
                        if rank == 4:
                            coef = np.reshape(coef, [nc, 4])
                            a, b, c, d = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3]
                            dat_i[kk, :] = d + (c + (b + a * dxx_k) * dxx_k) * dxx_k
                        elif rank == 2:
                            coef = np.reshape(coef, [nc, 2])
                            a, b = coef[:, 0], coef[:, 1]
                            dat_i[kk, :] = b + a * dxx_k
                    dats.append(dat_i)
                dic_val[key] = dats
        return dic_val

    def mapping2(self, x, dic_map, cfgs, rank=4):
        r"""Evaluate value by mapping table of numpy."""
        tf.reset_default_graph()
        t_x = tf.placeholder(tf.float64, [None, 1], "t_x")
        t_table = tf.placeholder(tf.float64, [None, None], "t_table")
        t_table_grad = tf.placeholder(tf.float64, [None, None], "t_table_grad")
        t_table_info = tf.placeholder(tf.float64, [None], "t_table_info")
        t_y = op_module.map_flt_nvnmd(t_x, t_table, t_table_grad, t_table_info)
        sess = get_sess()
        #
        n = len(x)
        dic_val = {}
        for key in dic_map.keys():
            val = dic_map[key]
            if isinstance(val, list):
                dats = []
                for ii in range(len(val)):
                    val_i = val[ii]
                    feed_dict = {
                        t_x: x,
                        t_table: val_i,
                        t_table_grad: val_i * 0.0,
                        t_table_info: np.reshape(np.array(cfgs), [-1]),
                    }
                    dat_i = run_sess(sess, t_y, feed_dict=feed_dict)
                    dat_i = np.reshape(dat_i, [n, -1])
                    dats.append(dat_i)
                dic_val[key] = dats
        return dic_val

    def plot_lines(self, x, dic1, dic2=None):
        r"""Plot lines to see accuracy."""
        for key in dic1.keys():
            val1 = dic1[key]
            if dic2 is None:
                val2 = dic1[key]
            else:
                val2 = dic2[key]
            #
            if isinstance(val1, list):
                for ii in range(len(val1)):
                    val1_i = val1[ii]
                    val2_i = val2[ii]
                    nc = np.shape(val1_i)[1]

    def build_map_coef(self, cfgs, x, ys, grads, grad_grads, Nr, Nc, rank=4):
        r"""Build mapping table coefficient
        cfgs: cfg list
        cfg = x0, x1, dx.

        coef2:
        a x + b = y
        / b = y0
        \ a = (y1 - y0) / L

        coef4:
        a x^3 + b x^2 + c x + d = y:
        / d = y0
        | c = y0'
        | b = (3 y1 - dx dy' - 2dx y0' - 3y0) / dx^2
        \ a = (dx y1' - 2 y1 + dx y0' + 2 y0) / dx^3
        """

        def cal_coef2(cfg, x, y, dy):
            x = np.reshape(x, [-1])
            coefs = []
            for cfg in cfgs:
                x0, x1, dx, N0, N1 = cfg
                Nd = N1 - N0
                idx = np.logical_and(
                    x >= x0,
                    x <= x1,
                    np.abs((x - x0) - np.floor((x - x0) / dx) * dx) < 1e-4,
                )
                y0 = y[idx][:-1]
                y1 = y[idx][1:]
                y0 = y0[:Nd]
                y1 = y1[:Nd]
                a = (y1 - y0) / dx
                b = y0
                coef = np.concatenate([a, b])
                coef = np.transpose(np.reshape(coef, [2, -1]))
                coefs.append(coef)
            coefs = np.concatenate(coefs)
            return coefs

        def cal_coef4(cfg, x, y, dy):
            x = np.reshape(x, [-1])
            coefs = []
            for cfg in cfgs:
                x0, x1, dx, N0, N1 = cfg
                Nd = N1 - N0
                diff_x = np.abs((x - x0) - np.round((x - x0) / dx) * dx)
                idx = np.logical_and(np.logical_and(x >= x0, x <= x1), diff_x < 1.0e-4)
                y0 = y[idx][:-1]
                y1 = y[idx][1:]
                dy0 = dy[idx][:-1]
                dy1 = dy[idx][1:]
                y0 = y0[:Nd]
                y1 = y1[:Nd]
                dy0 = dy0[:Nd]
                dy1 = dy1[:Nd]
                #
                a = (dx * dy1 - 2 * y1 + dx * dy0 + 2 * y0) / dx**3
                b = (3 * y1 - dx * dy1 - 2 * dx * dy0 - 3 * y0) / dx**2
                c = dy0
                d = y0
                coef = np.concatenate([a, b, c, d])
                coef = np.transpose(np.reshape(coef, [4, -1]))
                coefs.append(coef)
            coefs = np.concatenate(coefs)
            return coefs

        #
        cal_coef = cal_coef4 if (rank == 4) else cal_coef2
        coefs = []
        coef_grads = []
        for ii in range(Nr):
            y_i = ys[ii]
            grad_i = grads[ii]
            grad_grad_i = grad_grads[ii]
            #
            coef_i = []
            coef_grad_i = []
            for jj in range(Nc):
                y_ij = y_i[:, jj]
                grad_ij = grad_i[:, jj]
                grad_grad_ij = grad_grad_i[:, jj]
                coef_ij = cal_coef(cfgs, x, y_ij, grad_ij)
                coef_grad_ij = cal_coef(cfgs, x, grad_ij, grad_grad_ij)
                coef_i.append(coef_ij)
                coef_grad_i.append(coef_grad_ij)
            coef_i = np.concatenate(coef_i, axis=1)
            coef_grad_i = np.concatenate(coef_grad_i, axis=1)
            coefs.append(coef_i)
            coef_grads.append(coef_grad_i)
        return coefs, coef_grads

    def build_grad(self, x, y, Nr, Nc):
        r""": Build gradient of tensor y of x."""
        grads = []
        grad_grads = []
        for ii in range(Nr):
            y_i = y[ii]
            grad_i = []
            grad_grad_i = []
            for jj in range(Nc):
                y_ij = y_i[:, jj]
                grad_ij = tf.gradients(y_ij, x)[0]
                grad_grad_ij = tf.gradients(grad_ij, x)[0]
                grad_i.append(grad_ij)
                grad_grad_i.append(grad_grad_ij)
            grad_i = tf.concat(grad_i, axis=1)
            grad_grad_i = tf.concat(grad_grad_i, axis=1)
            grads.append(grad_i)
            grad_grads.append(grad_grad_i)
        return grads, grad_grads

    def build_u2s(self, r2):
        r"""Build tensor s, s=s(r2)."""
        rmin = nvnmd_cfg.dscp["rcut_smth"]
        rmax = nvnmd_cfg.dscp["rcut"]
        ntype = nvnmd_cfg.dscp["ntype"]

        if "train_attr.min_nbor_dist" in nvnmd_cfg.weight.keys():
            min_dist = nvnmd_cfg.weight["train_attr.min_nbor_dist"]
        else:
            min_dist = rmin
        min_dist = 0.5 if (min_dist > 0.5) else (min_dist - 0.1)
        #
        avg, std = get_normalize(nvnmd_cfg.weight)
        avg, std = np.float64(avg), np.float64(std)
        r = tf.sqrt(r2)
        r_ = tf.clip_by_value(r, rmin, rmax)
        r__ = tf.clip_by_value(r, min_dist, rmax)
        uu = (r_ - rmin) / (rmax - rmin)
        vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1

        sl = []
        hl = []

        for tt in range(ntype):
            s = vv / r__
            h = s / r__
            s = tf.reshape(s, [-1, 1])
            h = tf.reshape(h, [-1, 1])
            s = (s - avg[tt, 0]) / std[tt, 0]
            h = h / std[tt, 1]
            sl.append(s)
            hl.append(h)
        return sl, hl

    def build_u2s_grad(self):
        r"""Build gradient of s with respect to u (r^2)."""
        ntype = nvnmd_cfg.dscp["ntype"]
        #
        dic_ph = {}
        dic_ph["u"] = tf.placeholder(tf.float64, [None, 1], "t_u")
        dic_ph["s"], dic_ph["h"] = self.build_u2s(dic_ph["u"])
        dic_ph["s_grad"], dic_ph["s_grad_grad"] = self.build_grad(
            dic_ph["u"], dic_ph["s"], ntype, 1
        )
        dic_ph["h_grad"], dic_ph["h_grad_grad"] = self.build_grad(
            dic_ph["u"], dic_ph["h"], ntype, 1
        )
        return dic_ph

    def run_u2s(self):
        r"""Build u->s graph and run it to get value of mapping table."""
        # ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp["ntype"]
        avg, std = get_normalize(nvnmd_cfg.weight)
        avg, std = np.float64(avg), np.float64(std)
        rc_max = nvnmd_cfg.dscp["rc_max"]

        tf.reset_default_graph()
        dic_ph = self.build_u2s_grad()
        sess = get_sess()

        # N = NUM_MAPT
        N = 512
        N2 = int(rc_max**2)
        # N+1 ranther than N for calculating defference
        keys = list(dic_ph.keys())
        vals = list(dic_ph.values())

        u = N2 * np.reshape(np.arange(0, N + 1) / N, [-1, 1])
        res_lst = run_sess(sess, vals, feed_dict={dic_ph["u"]: u})
        res_dic = dict(zip(keys, res_lst))

        u2 = N2 * np.reshape(np.arange(0, N * 16 + 1) / (N * 16), [-1, 1])
        res_lst2 = run_sess(sess, vals, feed_dict={dic_ph["u"]: u2})
        res_dic2 = dict(zip(keys, res_lst2))  # reference for commpare

        # change value
        for tt in range(ntype):
            res_dic["s"][tt][0] = -avg[tt, 0] / std[tt, 0]
            res_dic["s_grad"][tt][0] = 0
            res_dic["s_grad_grad"][tt][0] = 0
            res_dic["h"][tt][0] = 0
            res_dic["h_grad"][tt][0] = 0
            res_dic["h_grad_grad"][tt][0] = 0
            #
            res_dic2["s"][tt][0] = -avg[tt, 0] / std[tt, 0]
            res_dic2["s_grad"][tt][0] = 0
            res_dic2["s_grad_grad"][tt][0] = 0
            res_dic2["h"][tt][0] = 0
            res_dic2["h_grad"][tt][0] = 0
            res_dic2["h_grad_grad"][tt][0] = 0

        sess.close()
        return res_dic, res_dic2

    def build_s2g(self, s):
        r"""Build s->G
        s is switch function
        G is embedding net output.
        """
        ntypex = nvnmd_cfg.dscp["ntypex"]
        ntype = nvnmd_cfg.dscp["ntype"]

        activation_fn = tf.tanh
        outputs_size = nvnmd_cfg.dscp["NNODE_FEAS"]

        xyz_scatters = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                xyz_scatter = s
                for ll in range(1, len(outputs_size)):
                    w, b = get_filter_weight(nvnmd_cfg.weight, tt, tt2, ll)
                    w, b = np.float64(w), np.float64(b)
                    if outputs_size[ll] == outputs_size[ll - 1]:
                        xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
                    elif outputs_size[ll] == outputs_size[ll - 1] * 2:
                        xyz_scatter = tf.concat(
                            [xyz_scatter, xyz_scatter], 1
                        ) + activation_fn(tf.matmul(xyz_scatter, w) + b)
                    else:
                        xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
                xyz_scatters.append(xyz_scatter)
        return xyz_scatters

    def build_s2g_grad(self):
        r"""Build gradient of G with respect to s."""
        ntypex = nvnmd_cfg.dscp["ntypex"]
        ntype = nvnmd_cfg.dscp["ntype"]
        M1 = nvnmd_cfg.dscp["M1"]
        #
        dic_ph = {}
        dic_ph["s"] = tf.placeholder(tf.float64, [None, 1], "t_s")
        dic_ph["g"] = self.build_s2g(dic_ph["s"])
        dic_ph["g_grad"], dic_ph["g_grad_grad"] = self.build_grad(
            dic_ph["s"], dic_ph["g"], ntypex * ntype, M1
        )
        return dic_ph

    def run_s2g(self):
        r"""Build s-> graph and run it to get value of mapping table."""
        smin = nvnmd_cfg.dscp["smin"]
        smax = nvnmd_cfg.dscp["smax"]

        tf.reset_default_graph()
        dic_ph = self.build_s2g_grad()
        sess = get_sess()

        N = 4096
        N2 = 16
        log.info(f"the range of s is [{smin}, {smax}]")
        # check
        if (smax - smin) > 16.0:
            log.warning("the range of s is over the limit (smax - smin) > 16.0")
        prec = N / N2
        smin_ = np.floor(smin * prec - 1) / prec
        #
        keys = list(dic_ph.keys())
        vals = list(dic_ph.values())

        s = N2 * np.reshape(np.arange(0, N + 1) / N, [-1, 1]) + smin_
        res_lst = run_sess(sess, vals, feed_dict={dic_ph["s"]: s})
        res_dic = dict(zip(keys, res_lst))

        s2 = N2 * np.reshape(np.arange(0, N * 16 + 1) / (N * 16), [-1, 1]) + smin_
        res_lst2 = run_sess(sess, vals, feed_dict={dic_ph["s"]: s2})
        res_dic2 = dict(zip(keys, res_lst2))

        sess.close()
        return res_dic, res_dic2


def mapt(
    *,
    nvnmd_config: Optional[str] = "nvnmd/config.npy",
    nvnmd_weight: Optional[str] = "nvnmd/weight.npy",
    nvnmd_map: Optional[str] = "nvnmd/map.npy",
    **kwargs,
):
    # build mapping table
    mapObj = MapTable(nvnmd_config, nvnmd_weight, nvnmd_map)
    mapObj.build_map()
