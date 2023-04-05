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
from deepmd.nvnmd.utils.encode import (
    Encode,
)
from deepmd.nvnmd.utils.fio import (
    FioBin,
    FioTxt,
)
from deepmd.nvnmd.utils.network import (
    get_sess,
)
from deepmd.nvnmd.utils.weight import (
    get_fitnet_weight,
)
from deepmd.utils.sess import (
    run_sess,
)

log = logging.getLogger(__name__)


class Wrap:
    r"""Generate the binary model file (model.pb).

    the model file can be use to run the NVNMD with lammps
    the pair style need set as:

    .. code-block:: lammps

        pair_style nvnmd model.pb
        pair_coeff * *

    Parameters
    ----------
    config_file
        input file name
        an .npy file containing the configuration information of NVNMD model
    weight_file
        input file name
        an .npy file containing the weights of NVNMD model
    map_file
        input file name
        an .npy file containing the mapping tables of NVNMD model
    model_file
        output file name
        an .pb file containing the model using in the NVNMD

    References
    ----------
    DOI: 10.1038/s41524-022-00773-z
    """

    def __init__(
        self, config_file: str, weight_file: str, map_file: str, model_file: str
    ):
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file
        self.model_file = model_file

        jdata = jdata_deepmd_input["nvnmd"]
        jdata["config_file"] = config_file
        jdata["weight_file"] = weight_file
        jdata["map_file"] = map_file
        jdata["enable"] = True

        nvnmd_cfg.init_from_jdata(jdata)

    def wrap(self):
        dscp = nvnmd_cfg.dscp
        ctrl = nvnmd_cfg.ctrl

        M1 = dscp["M1"]
        ntype = dscp["ntype"]
        ntype_max = dscp["ntype_max"]
        NSTDM_M1X = ctrl["NSTDM_M1X"]
        e = Encode()

        bcfg = self.wrap_dscp()
        bfps, bbps = self.wrap_fitn()
        bswt, bdsw, bfea, bgra = self.wrap_map()

        # split data with {nbit} bits per row
        hcfg = e.bin2hex(e.split_bin(bcfg, 72))
        # the data must bigger than 128
        hcfg = e.extend_list(hcfg, 128 if len(hcfg) < 128 else len(hcfg))

        # network weight
        hfps = e.bin2hex(e.split_bin(bfps, 72))
        hbps = e.bin2hex(e.split_bin(bbps, 72))

        # extend the data from ntype into ntype_max
        hswt = e.bin2hex(bswt)
        hdsw = e.bin2hex(bdsw)
        hfea = e.bin2hex(bfea)
        hgra = e.bin2hex(bgra)
        # extend data according to the number of bits per row of BRAM
        nhex = 32
        datas = [hcfg, hfps, hbps, hswt, hdsw, hfea, hgra]
        keys = "cfg fps bps swt dsw fea gra".split()
        nhs = []
        nws = []
        for ii in range(len(datas)):
            k = keys[ii]
            d = datas[ii]
            h = len(d)
            w = len(d[0])
            nhs.append(h)
            nws.append(w)  # nhex * 4 // 8 = nbyte
            #
            w_full = np.ceil(w * 4 / nhex) * nhex  # 32 bit per data
            d = e.extend_hex(d, w_full)
            # DEVELOP_DEBUG
            if jdata_sys["debug"]:
                log.info("%s: %d x % d bit" % (k, h, w * 4))
                FioTxt().save("nvnmd/wrap/h%s.txt" % (k), d)
            datas[ii] = d
        #
        nvnmd_cfg.size["NH_DATA"] = nhs
        nvnmd_cfg.size["NW_DATA"] = nws
        nvnmd_cfg.save(nvnmd_cfg.config_file)
        head = self.wrap_head(nhs, nws)
        #
        hs = [] + head
        for d in datas:
            hs.extend(d)

        FioBin().save(self.model_file, hs)
        log.info("NVNMD: finish wrapping model file")

    def wrap_head(self, nhs, nws):
        nbit = nvnmd_cfg.nbit
        NBIT_MODEL_HEAD = nbit["NBIT_MODEL_HEAD"]
        NBIT_FIXD_FL = nbit["NBIT_FIXD_FL"]
        rcut = nvnmd_cfg.dscp["rcut"]

        bs = ""
        e = Encode()
        # height
        for n in nhs:
            bs = e.dec2bin(n, NBIT_MODEL_HEAD)[0] + bs
        # weight
        for n in nws:
            bs = e.dec2bin(n, NBIT_MODEL_HEAD)[0] + bs
        # dscp
        RCUT = e.qr(rcut, NBIT_FIXD_FL)
        bs = e.dec2bin(RCUT, NBIT_MODEL_HEAD)[0] + bs
        # extend
        hs = e.bin2hex(bs)
        hs = e.extend_hex(hs, NBIT_MODEL_HEAD * 32)
        return hs

    def wrap_dscp(self):
        r"""Wrap the configuration of descriptor.

        [NBIT_IDX_S2G-1:0] SHIFT_IDX_S2G
        [NBIT_NEIB*NTYPE-1:0] SELs
        [NBIT_FIXD*M1*NTYPE*NTYPE-1:0] GSs
        [NBIT_FLTE-1:0] NEXPO_DIV_NI

        """
        dscp = nvnmd_cfg.dscp
        nbit = nvnmd_cfg.nbit
        mapt = nvnmd_cfg.map
        NBIT_IDX_S2G = nbit["NBIT_IDX_S2G"]
        NBIT_NEIB = nbit["NBIT_NEIB"]
        NBIT_FLTE = nbit["NBIT_FLTE"]
        NBIT_FIXD = nbit["NBIT_FIXD"]
        NBIT_FIXD_FL = nbit["NBIT_FIXD_FL"]
        M1 = dscp["M1"]
        ntype = dscp["ntype"]
        ntype_max = dscp["ntype_max"]

        bs = ""
        e = Encode()
        # shift_idx_s2g
        x_st, x_ed, x_dt, N0, N1 = mapt["cfg_s2g"][0]
        shift_idx_s2g = int(np.round(-x_st / x_dt))
        bs = e.dec2bin(shift_idx_s2g, NBIT_IDX_S2G)[0] + bs
        # sel
        SEL = dscp["SEL"]
        bs = e.dec2bin(SEL[0], NBIT_NEIB)[0] + bs
        bs = e.dec2bin(SEL[1], NBIT_NEIB)[0] + bs
        bs = e.dec2bin(SEL[2], NBIT_NEIB)[0] + bs
        bs = e.dec2bin(SEL[3], NBIT_NEIB)[0] + bs
        # GS
        tf.reset_default_graph()
        t_x = tf.placeholder(tf.float64, [None, 1], "t_x")
        t_table = tf.placeholder(tf.float64, [None, None], "t_table")
        t_table_grad = tf.placeholder(tf.float64, [None, None], "t_table_grad")
        t_table_info = tf.placeholder(tf.float64, [None], "t_table_info")
        t_y = op_module.map_flt_nvnmd(t_x, t_table, t_table_grad, t_table_info)
        sess = get_sess()
        #
        GSs = []
        for tt in range(ntype_max):
            for tt2 in range(ntype_max):
                if (tt < ntype) and (tt2 < ntype):
                    # s
                    mi = mapt["s"][tt]
                    cfgs = mapt["cfg_u2s"]
                    cfgs = np.array([np.float64(v) for vs in cfgs for v in vs])
                    feed_dict = {
                        t_x: np.ones([1, 1]) * 0.0,
                        t_table: mi,
                        t_table_grad: mi * 0.0,
                        t_table_info: cfgs,
                    }
                    si = run_sess(sess, t_y, feed_dict=feed_dict)
                    si = np.reshape(si, [-1])[0]
                    # G
                    mi = mapt["g"][tt2]
                    cfgs = mapt["cfg_s2g"]
                    cfgs = np.array([np.float64(v) for vs in cfgs for v in vs])
                    feed_dict = {
                        t_x: np.ones([1, 1]) * si,
                        t_table: mi,
                        t_table_grad: mi * 0.0,
                        t_table_info: cfgs,
                    }
                    gi = run_sess(sess, t_y, feed_dict=feed_dict)
                    gsi = np.reshape(si, [-1]) * np.reshape(gi, [-1])
                else:
                    gsi = np.zeros(M1)
                for ii in range(M1):
                    GSs.extend(e.dec2bin(e.qr(gsi[ii], NBIT_FIXD_FL), NBIT_FIXD, True))
        sGSs = "".join(GSs[::-1])
        bs = sGSs + bs
        #
        NIX = dscp["NIX"]
        ln2_NIX = -int(np.log2(NIX))
        bs = e.dec2bin(ln2_NIX, NBIT_FLTE, signed=True)[0] + bs
        return bs

    def wrap_fitn(self):
        r"""Wrap the weights of fitting net."""
        dscp = nvnmd_cfg.dscp
        fitn = nvnmd_cfg.fitn
        weight = nvnmd_cfg.weight
        nbit = nvnmd_cfg.nbit
        ctrl = nvnmd_cfg.ctrl

        ntype = dscp["ntype"]
        ntype_max = dscp["ntype_max"]
        nlayer_fit = fitn["nlayer_fit"]
        NNODE_FITS = fitn["NNODE_FITS"]

        NBIT_FIT_DATA = nbit["NBIT_FIT_DATA"]
        NBIT_FIT_DATA_FL = nbit["NBIT_FIT_DATA_FL"]
        NBIT_FIT_WEIGHT = nbit["NBIT_FIT_WEIGHT"]
        NBIT_FIT_DISP = nbit["NBIT_FIT_DISP"]
        NBIT_FIT_WXDB = nbit["NBIT_FIT_WXDB"]
        NSTDM = ctrl["NSTDM"]
        NSEL = ctrl["NSEL"]

        # encode all parameters
        bb, bdr, bdc, bwr, bwc = [], [], [], [], []
        for ll in range(nlayer_fit):
            bbt, bdrt, bdct, bwrt, bwct = [], [], [], [], []
            for tt in range(ntype_max):
                # get parameters: weight and bias
                if tt < ntype:
                    w, b = get_fitnet_weight(weight, tt, ll, nlayer_fit)
                else:
                    w, b = get_fitnet_weight(weight, 0, ll, nlayer_fit)
                    w = w * 0
                    b = b * 0
                # restrict the shift value of energy
                if ll == (nlayer_fit - 1):
                    b = b * 0
                bbi = self.wrap_bias(b, NBIT_FIT_WXDB, NBIT_FIT_DATA_FL)
                bdri, bdci, bwri, bwci = self.wrap_weight(
                    w, NBIT_FIT_DISP, NBIT_FIT_WEIGHT
                )
                bbt.append(bbi)
                bdrt.append(bdri)
                bdct.append(bdci)
                bwrt.append(bwri)
                bwct.append(bwci)
            bb.append(bbt)
            bdr.append(bdrt)
            bdc.append(bdct)
            bwr.append(bwrt)
            bwc.append(bwct)
        #
        bfps, bbps = [], []
        for ss in range(NSEL):
            tt = ss // NSTDM
            sc = ss % NSTDM
            sr = ss % NSTDM
            bfp, bbp = [], []
            for ll in range(nlayer_fit):
                nr = NNODE_FITS[ll]
                nc = NNODE_FITS[ll + 1]
                nrs = int(np.ceil(nr / NSTDM))
                ncs = int(np.ceil(nc / NSTDM))
                if nc == 1:
                    # fp
                    bfp += [
                        bwc[ll][tt][sr * nrs + rr][cc]
                        for rr in range(nrs)
                        for cc in range(nc)
                    ]
                    bfp += [bdc[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                    bfp += [bb[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                    # bp
                    bbp += [
                        bwc[ll][tt][sr * nrs + rr][cc]
                        for rr in range(nrs)
                        for cc in range(nc)
                    ]
                    bbp += [bdc[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                    bbp += [bb[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                else:
                    # fp
                    bfp += [
                        bwc[ll][tt][rr][sc * ncs + cc]
                        for cc in range(ncs)
                        for rr in range(nr)
                    ]
                    bfp += [bdc[ll][tt][sc * ncs + cc] for cc in range(ncs)]
                    bfp += [bb[ll][tt][sc * ncs + cc] for cc in range(ncs)]
                    # bp
                    bbp += [
                        bwr[ll][tt][sr * nrs + rr][cc]
                        for rr in range(nrs)
                        for cc in range(nc)
                    ]
                    bbp += [bdr[ll][tt][sc * ncs + cc] for cc in range(ncs)]
                    bbp += [bb[ll][tt][sc * ncs + cc] for cc in range(ncs)]
            bfps.append("".join(bfp[::-1]))
            bbps.append("".join(bbp[::-1]))
        return bfps, bbps

    def wrap_bias(self, bias, NBIT_DATA, NBIT_DATA_FL):
        e = Encode()
        bias = e.qr(bias, NBIT_DATA_FL)
        Bs = e.dec2bin(bias, NBIT_DATA, True)
        return Bs

    def wrap_weight(self, weight, NBIT_DISP, NBIT_WEIGHT):
        r"""weight: weights of fittingNet
        NBIT_DISP: nbits of exponent of weight max value
        NBIT_WEIGHT: nbits of mantissa of weights.
        """
        NBIT_WEIGHT_FL = NBIT_WEIGHT - 2
        sh = weight.shape
        nr, nc = sh[0], sh[1]
        nrs = np.zeros(nr)
        ncs = np.zeros(nc)
        wrs = np.zeros([nr, nc])
        wcs = np.zeros([nr, nc])
        e = Encode()
        # row
        for ii in range(nr):
            wi = weight[ii, :]
            wi, expo_max = e.norm_expo(wi, NBIT_WEIGHT_FL, 0)
            nrs[ii] = expo_max
            wrs[ii, :] = wi
        # column
        for ii in range(nc):
            wi = weight[:, ii]
            wi, expo_max = e.norm_expo(wi, NBIT_WEIGHT_FL, 0)
            ncs[ii] = expo_max
            wcs[:, ii] = wi
        NRs = e.dec2bin(nrs, NBIT_DISP)
        NCs = e.dec2bin(ncs, NBIT_DISP)
        wrs = e.qr(wrs, NBIT_WEIGHT_FL)
        WRs = e.dec2bin(wrs, NBIT_WEIGHT, True)
        WRs = [[WRs[nc * rr + cc] for cc in range(nc)] for rr in range(nr)]
        wcs = e.qr(wcs, NBIT_WEIGHT_FL)
        WCs = e.dec2bin(wcs, NBIT_WEIGHT, True)
        WCs = [[WCs[nc * rr + cc] for cc in range(nc)] for rr in range(nr)]
        return NRs, NCs, WRs, WCs

    def wrap_map(self):
        r"""Wrap the mapping table of embedding network."""
        dscp = nvnmd_cfg.dscp
        maps = nvnmd_cfg.map
        nbit = nvnmd_cfg.nbit

        M1 = dscp["M1"]
        ntype = dscp["ntype"]
        ntype_max = dscp["ntype_max"]

        NBIT_FLTD = nbit["NBIT_FLTD"]
        NBIT_FLTE = nbit["NBIT_FLTE"]
        NBIT_FLTF = nbit["NBIT_FLTF"]

        e = Encode()
        # get mapt
        swts = []
        dsws = []
        feas = []
        gras = []
        for tt in range(ntype_max):
            if tt < ntype:
                swt = np.concatenate([maps["s"][tt], maps["h"][tt]], axis=1)
                dsw = np.concatenate([maps["s_grad"][tt], maps["h_grad"][tt]], axis=1)
                fea = maps["g"][tt]
                gra = maps["g_grad"][tt]
            else:
                swt = np.concatenate([maps["s"][0], maps["h"][0]], axis=1)
                dsw = np.concatenate([maps["s_grad"][0], maps["h_grad"][0]], axis=1)
                fea = maps["g"][0]
                gra = maps["g_grad"][0]
                swt *= 0
                dsw *= 0
                fea *= 0
                gra *= 0
            swts.append(swt.copy())
            dsws.append(dsw.copy())
            feas.append(fea.copy())
            gras.append(gra.copy())
        mapts = [swts, dsws, feas, gras]
        # reshape
        opt_uram = True  # for reduce uram resource version
        if opt_uram:
            nmerges = [2 * 2, 2 * 2, 4 * 2, 4 * 2]  # n*(4/2)
        else:
            nmerges = [2 * 4, 2 * 4, 4 * 4, 4 * 4]  # n*(4)
        bss = []
        for ii in range(4):
            d = mapts[ii]
            d = np.reshape(d, [ntype_max, -1, 4])
            if opt_uram:
                d1 = d[:, :, 0:2]
                d2 = d[:, :, 2:4]
                d = np.concatenate([d1, d2])
            #
            bs = e.flt2bin(d, NBIT_FLTE, NBIT_FLTF)
            bs = e.reverse_bin(bs, nmerges[ii])
            bs = e.merge_bin(bs, nmerges[ii])
            bss.append(bs)
        bswt, bdsw, bfea, bgra = bss
        return bswt, bdsw, bfea, bgra


def wrap(
    *,
    nvnmd_config: Optional[str] = "nvnmd/config.npy",
    nvnmd_weight: Optional[str] = "nvnmd/weight.npy",
    nvnmd_map: Optional[str] = "nvnmd/map.npy",
    nvnmd_model: Optional[str] = "nvnmd/model.pb",
    **kwargs,
):
    wrapObj = Wrap(nvnmd_config, nvnmd_weight, nvnmd_map, nvnmd_model)
    wrapObj.wrap()
