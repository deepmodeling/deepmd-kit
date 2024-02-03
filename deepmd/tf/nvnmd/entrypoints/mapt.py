# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
)

import numpy as np

from deepmd.tf.env import (
    op_module,
    tf,
)
from deepmd.tf.nvnmd.data.data import (
    jdata_deepmd_input_v0,
    jdata_sys,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.nvnmd.utils.fio import (
    FioDic,
)
from deepmd.tf.nvnmd.utils.network import (
    get_sess,
)
from deepmd.tf.nvnmd.utils.weight import (
    get_filter_type_weight,
    get_filter_weight,
    get_normalize,
    get_type_embedding_weight,
)
from deepmd.tf.utils.sess import (
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

        jdata = jdata_deepmd_input_v0["nvnmd"]
        jdata["config_file"] = config_file
        jdata["weight_file"] = weight_file
        jdata["enable"] = True

        # 0 : xyz_scatter = xyz_scatter * two_embd + xyz_scatter;
        # Gs + 1, Gt + 0
        # 1 : xyz_scatter = xyz_scatter * two_embd + two_embd   ;
        # Gs + 0, Gt + 1
        self.Gs_Gt_mode = 1

        nvnmd_cfg.init_from_jdata(jdata)

    def build_map(self):
        if self.Gs_Gt_mode == 0:
            self.shift_Gs = 1
            self.shift_Gt = 0
        if self.Gs_Gt_mode == 1:
            self.shift_Gs = 0
            self.shift_Gt = 1
        #
        M = nvnmd_cfg.dscp["M1"]
        if nvnmd_cfg.version == 0:
            ndim = nvnmd_cfg.dscp["ntype"]
        if nvnmd_cfg.version == 1:
            ndim = 1
        # calculate grid point
        dic_u2s, dic_u2s_ref = self.run_u2s()
        dic_s2g, dic_s2g_ref = self.run_s2g()
        if nvnmd_cfg.version == 1:
            dic_t2g = self.run_t2g()
            dic_std = self.build_davg_dstd()
        # build mapping table
        dic_map1 = {}
        ## u2s
        u = np.reshape(dic_u2s["u"], [-1])
        cfg_u2s = [[u[0], u[512], u[1] - u[0], 0, 512]]
        dic_map1["s"], dic_map1["s_grad"] = self.build_map_coef(
            cfg_u2s,
            u,
            dic_u2s["s"],
            dic_u2s["s_grad"],
            dic_u2s["s_grad_grad"],
            ndim,
            1,
        )
        dic_map1["h"], dic_map1["h_grad"] = self.build_map_coef(
            cfg_u2s,
            u,
            dic_u2s["h"],
            dic_u2s["h_grad"],
            dic_u2s["h_grad_grad"],
            ndim,
            1,
        )
        ## s2g
        dic_map2 = {}
        s = np.reshape(dic_s2g["s"], [-1])
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
            ndim,
            M,
        )
        if nvnmd_cfg.version == 1:
            ## t2g
            dic_map3 = {}
            dic_map3["t_ebd"] = dic_t2g["t_ebd"]
            dic_map3["gt"] = dic_t2g["gt"]
            ## davg and dstd
            dic_map4 = {}
            dic_map4["davg_opp"] = dic_std["davg_opp"]
            dic_map4["dstd_inv"] = dic_std["dstd_inv"]
        # run mapping to test
        if jdata_sys["debug"]:
            dic_u2s_prd = self.mapping2(dic_u2s_ref["u"], dic_map1, cfg_u2s)
            dic_s2g_prd = self.mapping2(dic_s2g_ref["s"], dic_map2, cfg_s2g)
            self.plot_lines(dic_u2s_ref["u"], dic_u2s_prd, dic_u2s_ref)
            self.plot_lines(dic_s2g_ref["s"], dic_s2g_prd, dic_s2g_ref)
        # save
        self.map = {}
        self.map["cfg_u2s"] = cfg_u2s
        self.map["cfg_s2g"] = cfg_s2g
        self.map.update(dic_map1)
        self.map.update(dic_map2)
        if nvnmd_cfg.version == 1:
            self.map.update(dic_map3)
            self.map.update(dic_map4)
        #
        FioDic().save(self.map_file, self.map)
        log.info("NVNMD: finish building mapping table")
        return self.map

    def mapping(self, x, dic_map, cfgs):
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
                    nc = np.shape(val_i)[1] // 4
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
                        coef = np.reshape(coef, [nc, 4])
                        a, b, c, d = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3]
                        dat_i[kk, :] = d + (c + (b + a * dxx_k) * dxx_k) * dxx_k
                    dats.append(dat_i)
                dic_val[key] = dats
        return dic_val

    def mapping2(self, x, dic_map, cfgs):
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
            is_list = isinstance(val, list)
            if not is_list:
                val = [val]
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
            if not is_list:
                dats = dats[0]
            dic_val[key] = dats
        return dic_val

    def plot_lines(self, x, dic1, dic2=None):
        r"""Plot lines to see accuracy."""
        pass

    def build_map_coef(self, cfgs, x, ys, grads, grad_grads, Nr, Nc):
        r"""Build mapping table coefficient
        cfgs: cfg list
        cfg = x0, x1, dx.

        coef4:
        a x^3 + b x^2 + c x + d = y:
        / d = y0
        | c = y0'
        | b = (3 y1 - dx dy' - 2dx y0' - 3y0) / dx^2
        \ a = (dx y1' - 2 y1 + dx y0' + 2 y0) / dx^3
        """
        coefs = []
        coef_grads = []
        is_list = isinstance(ys, list)
        for ii in range(Nr):
            if is_list:
                y_i = ys[ii]
                grad_i = grads[ii]
                grad_grad_i = grad_grads[ii]
            else:
                y_i = ys
                grad_i = grads
                grad_grad_i = grad_grads
            #
            coef_i = []
            coef_grad_i = []
            for jj in range(Nc):
                y_ij = y_i[:, jj]
                grad_ij = grad_i[:, jj]
                grad_grad_ij = grad_grad_i[:, jj]
                coef_ij = self.cal_coef4(cfgs, x, y_ij, grad_ij)
                coef_grad_ij = self.cal_coef4(cfgs, x, grad_ij, grad_grad_ij)
                coef_i.append(coef_ij)
                coef_grad_i.append(coef_grad_ij)
            coef_i = np.concatenate(coef_i, axis=1)
            coef_grad_i = np.concatenate(coef_grad_i, axis=1)
            coefs.append(coef_i)
            coef_grads.append(coef_grad_i)
        if not is_list:
            coefs = coefs[0]
            coef_grads = coef_grads[0]
        return coefs, coef_grads

    def cal_coef4(self, cfgs, x, y, dy):
        r"""Build mapping table coefficient for one line
        coef4:
        a x^3 + b x^2 + c x + d = y:
        / d = y0
        | c = y0'
        | b = (3 y1 - dx dy' - 2dx y0' - 3y0) / dx^2
        \ a = (dx y1' - 2 y1 + dx y0' + 2 y0) / dx^3.
        """
        x = np.reshape(x, [-1])
        coefs = []
        for cfg in cfgs:
            x0, x1, dx, N0, N1 = cfg
            Nd = N1 - N0
            diff_x = np.abs((x - x0) - np.round((x - x0) / dx) * dx)
            idx = np.logical_and(np.logical_and(x >= x0, x <= x1), diff_x < 1.0e-6)
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

    def build_grad(self, x, y, Nr, Nc):
        r""": Build gradient of tensor y of x."""
        is_list = isinstance(y, list)
        grads = []
        grad_grads = []
        for ii in range(Nr):
            y_i = y[ii] if is_list else y
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
        if not is_list:
            grads = grads[0]
            grad_grads = grad_grads[0]
        return grads, grad_grads

    def build_u2s(self, r2):
        r"""Build tensor s, s=s(r2)."""
        rmin = nvnmd_cfg.dscp["rcut_smth"]
        rmax = nvnmd_cfg.dscp["rcut"]
        dmin = nvnmd_cfg.dscp["dmin"]

        min_dist = rmin
        if "train_attr.min_nbor_dist" in nvnmd_cfg.weight.keys():
            min_dist = nvnmd_cfg.weight["train_attr.min_nbor_dist"]
        if dmin > 1e-6:
            min_dist = dmin
        min_dist = 0.5 if (min_dist > 0.5) else (min_dist - 0.1)
        #
        r = tf.sqrt(r2)
        r_ = tf.clip_by_value(r, rmin, rmax)
        r__ = tf.clip_by_value(r, min_dist, rmax)
        uu = (r_ - rmin) / (rmax - rmin)
        vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1

        if nvnmd_cfg.version == 0:
            ntype = nvnmd_cfg.dscp["ntype"]
            avg, std = get_normalize(nvnmd_cfg.weight)
            avg, std = np.float64(avg), np.float64(std)

            sl, hl = [], []
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

        if nvnmd_cfg.version == 1:
            s = vv / r__
            h = s / r__
            s = tf.reshape(s, [-1, 1])
            h = tf.reshape(h, [-1, 1])
            return [s], [h]

    def build_u2s_grad(self):
        r"""Build gradient of s with respect to u (r^2)."""
        if nvnmd_cfg.version == 0:
            ndim = nvnmd_cfg.dscp["ntype"]
        if nvnmd_cfg.version == 1:
            ndim = 1
        #
        dic_ph = {}
        dic_ph["u"] = tf.placeholder(tf.float64, [None, 1], "t_u")
        dic_ph["s"], dic_ph["h"] = self.build_u2s(dic_ph["u"])
        dic_ph["s_grad"], dic_ph["s_grad_grad"] = self.build_grad(
            dic_ph["u"], dic_ph["s"], ndim, 1
        )
        dic_ph["h_grad"], dic_ph["h_grad_grad"] = self.build_grad(
            dic_ph["u"], dic_ph["h"], ndim, 1
        )
        return dic_ph

    def run_u2s(self):
        r"""Build u->s graph and run it to get value of mapping table."""
        ntype = nvnmd_cfg.dscp["ntype"]
        if nvnmd_cfg.version == 0:
            ndim = nvnmd_cfg.dscp["ntype"]
        if nvnmd_cfg.version == 1:
            ndim = 1
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
        for tt in range(ndim):
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
        if nvnmd_cfg.version == 0:
            ntype = nvnmd_cfg.dscp["ntype"]
        if nvnmd_cfg.version == 1:
            ntype = 1
        #
        xyz_scatters = []
        for tt2 in range(ntype):
            wbs = [get_filter_weight(nvnmd_cfg.weight, tt2, ll) for ll in range(1, 5)]
            xyz_scatter = self.build_embedding_net(s, wbs)
            xyz_scatters.append(xyz_scatter)
        return xyz_scatters

    def build_s2g_grad(self):
        r"""Build gradient of G with respect to s."""
        M1 = nvnmd_cfg.dscp["M1"]
        #
        if nvnmd_cfg.version == 0:
            ntypex = nvnmd_cfg.dscp["ntypex"]
            ntype = nvnmd_cfg.dscp["ntype"]
            ndim = ntypex * ntype
            shift = 0
        if nvnmd_cfg.version == 1:
            ndim = 1
            shift = self.shift_Gs
        #
        dic_ph = {}
        dic_ph["s"] = tf.placeholder(tf.float64, [None, 1], "t_s")
        dic_ph["g"] = [g + shift for g in self.build_s2g(dic_ph["s"])]
        dic_ph["g_grad"], dic_ph["g_grad_grad"] = self.build_grad(
            dic_ph["s"], dic_ph["g"], ndim, M1
        )
        return dic_ph

    def run_s2g(self):
        r"""Build s-> graph and run it to get value of mapping table."""
        smin = nvnmd_cfg.dscp["smin"]
        smax = nvnmd_cfg.dscp["smax"]
        # fix the bug: if model initial mode is 'init_from_model',
        # we need dmin to calculate smin and smax in mapt.py
        if smin == -2:
            davg, dstd = get_normalize(nvnmd_cfg.weight)
            nvnmd_cfg.get_s_range(davg, dstd)
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
        # the lower limit of switch function
        if nvnmd_cfg.version == 0:
            smin_ = np.floor(smin * prec - 1) / prec
        if nvnmd_cfg.version == 1:
            smin_ = 0
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

    def build_t2g(self):
        r"""Build t->G
        t is chemical species of center atom and neighbor atom
        G is embedding net output of type.
        """
        ntype = nvnmd_cfg.dscp["ntype"]
        filter_precision = tf.float64
        types = tf.convert_to_tensor(list(range(ntype)), dtype=tf.int32)
        ebd_type = tf.cast(
            tf.one_hot(tf.cast(types, dtype=tf.int32), int(ntype)),
            filter_precision,
        )
        ebd_type = tf.reshape(ebd_type, [-1, ntype])
        # type -> type_embedding
        dic_ph = {}
        dic_ph["t_one_hot"] = ebd_type
        wbs = [get_type_embedding_weight(nvnmd_cfg.weight, ll) for ll in range(1, 5)]
        ebd_type = self.build_embedding_net(dic_ph["t_one_hot"], wbs, None)
        last_type = tf.cast(tf.zeros([1, ebd_type.shape[1]]), filter_precision)
        ebd_type = tf.concat([ebd_type, last_type], 0)
        dic_ph["t_ebd"] = ebd_type
        # type_embedding of i, j atoms -> two_side_type_embedding
        type_embedding = dic_ph["t_ebd"]
        padding_ntypes = type_embedding.shape[0]
        type_embedding_nei = tf.tile(
            tf.reshape(type_embedding, [1, padding_ntypes, -1]),
            [padding_ntypes, 1, 1],
        )  # (ntypes) * ntypes * Y
        type_embedding_center = tf.tile(
            tf.reshape(type_embedding, [padding_ntypes, 1, -1]),
            [1, padding_ntypes, 1],
        )  # ntypes * (ntypes) * Y
        two_side_type_embedding = tf.concat(
            [type_embedding_nei, type_embedding_center], -1
        )  # ntypes * ntypes * (Y+Y)
        two_side_type_embedding = tf.reshape(
            two_side_type_embedding,
            [-1, two_side_type_embedding.shape[-1]],
        )
        # see se_atten.py in dp
        wbs = [get_filter_type_weight(nvnmd_cfg.weight, ll) for ll in range(1, 5)]
        dic_ph["gt"] = (
            self.build_embedding_net(two_side_type_embedding, wbs) + self.shift_Gt
        )
        return dic_ph

    def run_t2g(self):
        r"""Build t-> graph and run it to get value of mapping table."""
        tf.reset_default_graph()
        dic_ph = self.build_t2g()
        sess = get_sess()
        #
        keys = list(dic_ph.keys())
        vals = list(dic_ph.values())
        #
        res_lst = run_sess(sess, vals, feed_dict={})
        res_dic = dict(zip(keys, res_lst))

        sess.close()
        return res_dic

    def build_embedding_net(self, xx, wbs, activation_fn=tf.tanh):
        for ll in range(len(wbs)):
            # weight and bias
            w, b, t = wbs[ll]
            if (w is None) or (b is None):
                break
            w, b = np.float64(w), np.float64(b)
            # layer
            if activation_fn is None:
                hidden = tf.matmul(xx, w) + b
            else:
                hidden = activation_fn(tf.matmul(xx, w) + b)
            # resnet
            shw = w.shape
            if shw[1] == shw[0]:
                if t is None:
                    xx += hidden
                else:
                    xx += hidden * t
            elif shw[1] == shw[0] * 2:
                if t is None:
                    xx = tf.concat([xx, xx], 1) + hidden
                else:
                    xx = tf.concat([xx, xx], 1) + hidden * t
            else:
                xx = hidden
        return xx

    def build_davg_dstd(self):
        ntype = nvnmd_cfg.dscp["ntype"]
        davg, dstd = get_normalize(nvnmd_cfg.weight)
        #
        res_dic = {}
        res_dic["davg_opp"] = np.array([-davg[tt, 0:4] for tt in range(ntype)])
        res_dic["dstd_inv"] = np.array([1.0 / dstd[tt, 0:4] for tt in range(ntype)])
        return res_dic


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
