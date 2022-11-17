
import numpy as np
import logging

from deepmd.env import tf
from deepmd.utils.sess import run_sess

from deepmd.nvnmd.utils.fio import FioDic
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_normalize, get_rng_s, get_filter_weight
from deepmd.nvnmd.utils.network import get_sess

from deepmd.nvnmd.data.data import jdata_deepmd_input

from typing import List, Optional

log = logging.getLogger(__name__)


class MapTable:
    r"""Generate the mapping table describing the relastionship of
    atomic distance, cutoff function, and embedding matrix.

    three mapping table will be built:

    | :math:`r^2_{ji} \rightarrow s_{ji}`
    | :math:`r^2_{ji} \rightarrow sr_{ji}`
    | :math:`r^2_{ji} \rightarrow \mathcal{G}_{ji}`

    where :math:`s_{ji}` is cut-off function,
    :math:`sr_{ji} = \frac{s(r_{ji})}{r_{ji}}`, and
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

    def __init__(
            self,
            config_file: str,
            weight_file: str,
            map_file: str
    ):
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file

        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['weight_file'] = weight_file
        jdata['enable'] = True

        nvnmd_cfg.init_from_jdata(jdata)
        # map_table = self.build_map()

    def qqq(self, dat, NBIT_FEA_FL, NBIT_FEA_X, is_set_zero=False):
        dat = dat if isinstance(dat, list) else [dat]
        prec = 2 ** NBIT_FEA_FL
        N = int(2 ** NBIT_FEA_X)
        #
        dat2 = []
        for ii in range(len(dat)):
            dati = dat[ii]
            vi = dati[:-1]  # i
            vi1 = dati[1:]  # i+1
            # v = vi + dvi * (r - ri)
            # ri = i * dt
            # dvi = v(i+1) / dt
            vi = np.round(vi * prec) / prec
            vi1 = np.round(vi1 * prec) / prec
            dvi = vi1 - vi
            if is_set_zero:
                dvi[0] = 0
            #
            v = [np.reshape(vp, [N, -1]) for vp in [vi, dvi]]
            dat2.append(v)
        return dat2

    def build_map(self):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        NBIT_FEA_FL = nvnmd_cfg.nbit['NBIT_FEA_FL']
        NBIT_FEA_X = nvnmd_cfg.nbit['NBIT_FEA_X']

        dic = self.run_u2s()
        dic.update(self.run_s2G(dic))

        # quantize s and G
        prec = 2**NBIT_FEA_FL
        for tt in range(ntypex):
            dic['s'][tt][0] = np.round(dic['s'][tt][0] * prec) / prec
            dic['sr'][tt][0] = np.round(dic['sr'][tt][0] * prec) / prec
            for tt2 in range(ntype):
                v = np.round(dic['G'][tt * ntype + tt2][0] * prec) / prec
                dic['G'][tt * ntype + tt2][0] = v

        maps = {}
        keys = 's,sr,ds_dr2,dsr_dr2,G,dG_ds'.split(',')
        keys2 = 'G,dG_ds'.split(',')
        for key in keys:
            val = self.qqq(dic[key], NBIT_FEA_FL, NBIT_FEA_X, key not in keys2)
            maps[key] = val

        N = int(2**NBIT_FEA_X)
        maps2 = {}
        maps2['r2'] = dic['r2'][0:N]
        maps2['s2'] = dic['s2'][0:N]
        for tt in range(ntypex):
            for tt2 in range(ntype):
                postfix = f'_t{tt}_t{tt2}'
                for key in keys:
                    maps2[key + postfix] = []
                    maps2[key + postfix].append(maps[key][tt * ntype + tt2][0].reshape([N, -1]))
                    maps2[key + postfix].append(maps[key][tt * ntype + tt2][1].reshape([N, -1]))
        self.map = maps2

        FioDic().save(self.map_file, self.map)
        log.info("NVNMD: finish building mapping table")
        return self.map

# =====================================================================
# build r2s
# =====================================================================

    def build_r2s(self, r2):
        # limit = nvnmd_cfg.dscp['rc_lim']
        rmin = nvnmd_cfg.dscp['rcut_smth']
        rmax = nvnmd_cfg.dscp['rcut']
        # ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        avg, std = get_normalize(nvnmd_cfg.weight)
        avg, std = np.float32(avg), np.float32(std)
        r = tf.sqrt(r2)
        r_ = tf.clip_by_value(r, rmin, rmax)
        r__ = tf.clip_by_value(r, 0, rmax)
        uu = (r_ - rmin) / (rmax - rmin)
        vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1

        sl = []
        srl = []

        for tt in range(ntype):
            s = vv / r__
            sr = s / r__
            s = tf.reshape(s, [-1, 1])
            sr = tf.reshape(sr, [-1, 1])
            s = (s - avg[tt, 0]) / std[tt, 0]
            sr = sr / std[tt, 1]
            sl.append(s)
            srl.append(sr)
        return sl, srl

    def build_ds_dr(self, r2, s, sr):
        # ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']

        ds_drl = []
        dsr_drl = []
        for tt in range(ntype):
            si = s[tt]
            sri = sr[tt]
            ds_dr = tf.gradients(si, r2)
            dsr_dr = tf.gradients(sri, r2)
            ds_drl.append(ds_dr[0])
            dsr_drl.append(dsr_dr[0])
        return ds_drl, dsr_drl

    def build_r2s_r2ds(self):
        dic_ph = {}
        dic_ph['r2'] = tf.placeholder(tf.float32, [None, 1], 't_r2')
        dic_ph['s'], dic_ph['sr'] = self.build_r2s(dic_ph['r2'])
        dic_ph['ds_dr2'], dic_ph['dsr_dr2'] = self.build_ds_dr(dic_ph['r2'], dic_ph['s'], dic_ph['sr'])

        return dic_ph

    def run_u2s(self):
        # ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        avg, std = get_normalize(nvnmd_cfg.weight)
        avg, std = np.float32(avg), np.float32(std)
        NBIT_FEA_X = nvnmd_cfg.nbit['NBIT_FEA_X']
        NBIT_FEA_X_FL = nvnmd_cfg.nbit['NBIT_FEA_X_FL']

        dic_ph = self.build_r2s_r2ds()
        sess = get_sess()

        N = 2 ** NBIT_FEA_X
        N2 = 2 ** NBIT_FEA_X_FL
        # N+1 ranther than N for calculating defference
        r2 = 1.0 * np.arange(0, N + 1) / N2
        r2 = np.reshape(r2, [-1, 1])
        feed_dic = {dic_ph['r2']: r2}
        key = 'r2,s,sr,ds_dr2,dsr_dr2'
        tlst = [dic_ph[k] for k in key.split(',')]
        # res = sess.run(tlst, feed_dic)
        res = run_sess(sess, tlst, feed_dict=feed_dic)

        res2 = {}
        key = key.split(',')
        for ii in range(len(key)):
            res2[key[ii]] = res[ii]

        # change value
        # set 0 value, when u=0
        for tt in range(ntype):
            res2['s'][tt][0] = -avg[tt, 0] / std[tt, 0]
            res2['sr'][tt][0] = 0
            res2['ds_dr2'][tt][0] = 0
            res2['dsr_dr2'][tt][0] = 0

        # r = np.sqrt(res2['r2'])
        sess.close()

        return res2
# =====================================================================
# build s2G
# =====================================================================

    def build_s2G(self, s):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']

        activation_fn = tf.tanh
        outputs_size = nvnmd_cfg.dscp['NNODE_FEAS']

        xyz_scatters = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                xyz_scatter = s
                for ll in range(1, len(outputs_size)):
                    w, b = get_filter_weight(nvnmd_cfg.weight, tt, tt2, ll)
                    w, b = np.float32(w), np.float32(b)
                    if outputs_size[ll] == outputs_size[ll - 1]:
                        xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
                    elif outputs_size[ll] == outputs_size[ll - 1] * 2:
                        xyz_scatter = tf.concat([xyz_scatter, xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
                    else:
                        xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
                xyz_scatters.append(xyz_scatter)
        return xyz_scatters

    def build_dG_ds(self, G, s):
        ntypex = nvnmd_cfg.dscp['ntypex']
        ntype = nvnmd_cfg.dscp['ntype']
        M1 = nvnmd_cfg.dscp['M1']

        dG_ds = []
        for tt in range(ntypex):
            for tt2 in range(ntype):
                Gi = G[tt * ntype + tt2]
                si = s

                dG_ds_i = []
                for ii in range(M1):
                    dG_ds_ii = tf.reshape(tf.gradients(Gi[:, ii], si), [-1, 1])
                    dG_ds_i.append(dG_ds_ii)
                dG_ds_i = tf.concat(dG_ds_i, axis=1)
                dG_ds.append(dG_ds_i)
        return dG_ds

    def build_s2G_s2dG(self):
        # ntypex = nvnmd_cfg.dscp['ntypex']
        dic_ph = {}
        dic_ph['s2'] = tf.placeholder(tf.float32, [None, 1], 't_s')
        dic_ph['G'] = self.build_s2G(dic_ph['s2'])
        dic_ph['dG_ds'] = self.build_dG_ds(dic_ph['G'], dic_ph['s2'])
        return dic_ph

    def run_s2G(self, dat):
        NBIT_FEA_FL = nvnmd_cfg.nbit['NBIT_FEA_FL']
        NBIT_FEA_X = nvnmd_cfg.nbit['NBIT_FEA_X']
        NBIT_FEA_X2_FL = nvnmd_cfg.nbit['NBIT_FEA_X2_FL']
        prec = 2 ** NBIT_FEA_FL

        dic_ph = self.build_s2G_s2dG()
        sess = get_sess()

        N = 2 ** NBIT_FEA_X
        N2 = 2 ** NBIT_FEA_X2_FL
        s_min, s_max = get_rng_s(nvnmd_cfg.weight)
        #
        if (s_min < -2.0) or (s_max > 14.0):
            log.warning(f"the range of s [{s_min}, {s_max}] is over the limit [-2.0, 14.0]")
        s_min = -2.0
        s = s_min + np.arange(0, N + 1) / N2
        s = np.reshape(s, [-1, 1])
        feed_dic = {dic_ph['s2']: s}

        feed_dic = {dic_ph['s2']: s}
        key = 's2,G,dG_ds'
        tlst = [dic_ph[k] for k in key.split(',')]
        # res = sess.run(tlst, feed_dic)
        res = run_sess(sess, tlst, feed_dict=feed_dic)

        res2 = {}
        key = key.split(',')
        for ii in range(len(key)):
            res2[key[ii]] = res[ii]

        sess.close()
        return res2


def mapt(
    *,
    nvnmd_config: Optional[str] = 'nvnmd/config.npy',
    nvnmd_weight: Optional[str] = 'nvnmd/weight.npy',
    nvnmd_map: Optional[str] = 'nvnmd/map.npy',
    **kwargs
):
    # build mapping table
    mapObj = MapTable(nvnmd_config, nvnmd_weight, nvnmd_map)
    mapObj.build_map()
