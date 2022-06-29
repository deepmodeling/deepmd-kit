
import numpy as np
import logging

from deepmd.nvnmd.utils.fio import FioBin, FioTxt
from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_fitnet_weight
from deepmd.nvnmd.utils.encode import Encode
from deepmd.nvnmd.utils.op import map_nvnmd

from deepmd.nvnmd.data.data import jdata_deepmd_input, jdata_sys
from typing import List, Optional

log = logging.getLogger(__name__)


class Wrap():
    r"""Generate the binary model file (model.pb)
    the model file can be use to run the NVNMD with lammps
    the pair style need set as:

    | :code:`pair_style nvnmd model.pb`
    | :code:`pair_coeff * *`

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
        self,
        config_file: str,
        weight_file: str,
        map_file: str,
        model_file: str
    ):
        self.config_file = config_file
        self.weight_file = weight_file
        self.map_file = map_file
        self.model_file = model_file

        jdata = jdata_deepmd_input['nvnmd']
        jdata['config_file'] = config_file
        jdata['weight_file'] = weight_file
        jdata['map_file'] = map_file
        jdata['enable'] = True

        nvnmd_cfg.init_from_jdata(jdata)

    def wrap(self):
        dscp = nvnmd_cfg.dscp
        ctrl = nvnmd_cfg.ctrl

        M1 = dscp['M1']
        ntype = dscp['ntype']
        ntype_max = dscp['ntype_max']
        NSTDM_M1X = ctrl['NSTDM_M1X']
        e = Encode()

        bcfg = self.wrap_dscp()
        bfps, bbps = self.wrap_fitn()
        bfea, bgra = self.wrap_map()

        # split data with {nbit} bits per row
        hcfg = e.bin2hex(e.split_bin(bcfg, 72))
        # the legnth of hcfg need to be the multiples of NSTDM_M1X
        hcfg = e.extend_list(hcfg, int(np.ceil(len(hcfg) / NSTDM_M1X)) * NSTDM_M1X)

        hfps = e.bin2hex(e.split_bin(bfps, 72))
        # hfps = e.extend_list(hfps, (len(hfps) // ntype) * ntype_max)

        hbps = e.bin2hex(e.split_bin(bbps, 72))
        # hbps = e.extend_list(hbps, (len(hbps) // ntype) * ntype_max)

        # split into multiple rows
        bfea = e.split_bin(bfea, len(bfea[0]) // NSTDM_M1X)
        # bfea = e.reverse_bin(bfea, NSTDM_M1X)
        # extend the number of lines
        hfea = e.bin2hex(bfea)
        hfea = e.extend_list(hfea, (len(hfea) // ntype) * ntype_max)

        # split into multiple rows
        bgra = e.split_bin(bgra, len(bgra[0]) // NSTDM_M1X)
        # bgra = e.reverse_bin(bgra, NSTDM_M1X)
        # extend the number of lines
        hgra = e.bin2hex(bgra)
        hgra = e.extend_list(hgra, (len(hgra) // ntype) * ntype_max)

        # extend data according to the number of bits per row of BRAM
        nhex = 512
        hcfg = e.extend_hex(hcfg, nhex)
        hfps = e.extend_hex(hfps, nhex)
        hbps = e.extend_hex(hbps, nhex)
        hfea = e.extend_hex(hfea, nhex)
        hgra = e.extend_hex(hgra, nhex)

        # DEVELOP_DEBUG
        if jdata_sys['debug']:
            log.info("len(hcfg): %d" % (len(hcfg)))
            log.info("len(hfps): %d" % (len(hfps)))
            log.info("len(hbps): %d" % (len(hbps)))
            log.info("len(hfea): %d" % (len(hfea)))
            log.info("len(hgra): %d" % (len(hgra)))
            #
            FioTxt().save('nvnmd/wrap/hcfg.txt', hcfg)
            FioTxt().save('nvnmd/wrap/hfps.txt', hfps)
            FioTxt().save('nvnmd/wrap/hbps.txt', hbps)
            FioTxt().save('nvnmd/wrap/hfea.txt', hfea)
            FioTxt().save('nvnmd/wrap/hgra.txt', hgra)
        #
        NCFG = len(hcfg)
        NNET = len(hfps)
        NFEA = len(hfea)
        nvnmd_cfg.nbit['NCFG'] = NCFG
        nvnmd_cfg.nbit['NNET'] = NNET
        nvnmd_cfg.nbit['NFEA'] = NFEA
        nvnmd_cfg.save(nvnmd_cfg.config_file)
        head = self.wrap_head(NCFG, NNET, NFEA)
        #
        hs = [] + head
        hs.extend(hcfg)
        hs.extend(hfps)
        hs.extend(hbps)
        hs.extend(hfea)
        hs.extend(hgra)

        FioBin().save(self.model_file, hs)
        log.info("NVNMD: finish wrapping model file")

    def wrap_head(self, NCFG, NNET, NFEA):
        nbit = nvnmd_cfg.nbit
        NBTI_MODEL_HEAD = nbit['NBTI_MODEL_HEAD']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        rcut = nvnmd_cfg.dscp['rcut']

        bs = ''
        e = Encode()
        # nline
        bs = e.dec2bin(NCFG, NBTI_MODEL_HEAD)[0] + bs
        bs = e.dec2bin(NNET, NBTI_MODEL_HEAD)[0] + bs
        bs = e.dec2bin(NFEA, NBTI_MODEL_HEAD)[0] + bs
        # dscp
        RCUT = e.qr(rcut, NBIT_DATA_FL)
        bs = e.dec2bin(RCUT, NBTI_MODEL_HEAD)[0] + bs
        # extend
        hs = e.bin2hex(bs)
        nhex = 512
        hs = e.extend_hex(hs, nhex)
        return hs

    def wrap_dscp(self):
        r"""Wrap the configuration of descriptor
        """
        dscp = nvnmd_cfg.dscp
        nbit = nvnmd_cfg.nbit
        maps = nvnmd_cfg.map
        NBIT_FEA_X = nbit['NBIT_FEA_X']
        NBIT_FEA_X_FL = nbit['NBIT_FEA_X_FL']
        NBIT_FEA_X2_FL = nbit['NBIT_FEA_X2_FL']
        NBIT_FEA_FL = nbit['NBIT_FEA_FL']
        NBIT_LST = nbit['NBIT_LST']
        NBIT_SHIFT = nbit['NBIT_SHIFT']

        bs = ''
        e = Encode()
        # sel
        SEL = dscp['SEL']
        bs = e.dec2bin(SEL[0], NBIT_LST)[0] + bs
        bs = e.dec2bin(SEL[1], NBIT_LST)[0] + bs
        bs = e.dec2bin(SEL[2], NBIT_LST)[0] + bs
        bs = e.dec2bin(SEL[3], NBIT_LST)[0] + bs
        #
        NIX = dscp['NIX']
        ln2_NIX = int(np.log2(NIX))
        bs = e.dec2bin(ln2_NIX, NBIT_SHIFT)[0] + bs
        # G*s
        # ntypex = dscp['ntypex']
        ntype = dscp['ntype']
        # ntypex_max = dscp['ntypex_max']
        ntype_max = dscp['ntype_max']
        M1 = dscp['M1']
        GSs = []
        for tt in range(ntype_max):
            for tt2 in range(ntype_max):
                if (tt < ntype) and (tt2 < ntype):
                    s = maps[f's_t{0}_t{tt}'][0][0]
                    s = e.qf(s, NBIT_FEA_FL) / (2**NBIT_FEA_FL)
                    s_min = -2.0
                    yk, dyk = maps[f'G_t{0}_t{tt2}']
                    prec = 1 / (2 ** NBIT_FEA_X2_FL)
                    G = map_nvnmd(s - s_min, yk, dyk / prec, prec)
                    G = e.qf(G, NBIT_FEA_FL) / (2**NBIT_FEA_FL)
                    v = s * G
                else:
                    v = np.zeros(M1)
                for ii in range(M1):
                    GSs.extend(e.dec2bin(e.qr(v[ii], 2 * NBIT_FEA_FL), 27, True))
        sGSs = ''.join(GSs[::-1])
        bs = sGSs + bs
        return bs

    def wrap_fitn(self):
        r"""Wrap the weights of fitting net
        """
        dscp = nvnmd_cfg.dscp
        fitn = nvnmd_cfg.fitn
        weight = nvnmd_cfg.weight
        nbit = nvnmd_cfg.nbit
        ctrl = nvnmd_cfg.ctrl

        ntype = dscp['ntype']
        ntype_max = dscp['ntype_max']
        nlayer_fit = fitn['nlayer_fit']
        NNODE_FITS = fitn['NNODE_FITS']
        NBIT_SUM = nbit['NBIT_SUM']
        NBIT_DATA_FL = nbit['NBIT_DATA_FL']
        NBIT_WEIGHT = nbit['NBIT_WEIGHT']
        NBIT_WEIGHT_FL = nbit['NBIT_WEIGHT_FL']
        NBIT_SPE = nbit['NBIT_SPE']
        NSTDM = ctrl['NSTDM']
        NSEL = ctrl['NSEL']

        # encode all parameters
        bb, bw = [], []
        for ll in range(nlayer_fit):
            bbt, bwt = [], []
            for tt in range(ntype_max):
                # get parameters: weight and bias
                if (tt < ntype):
                    w, b = get_fitnet_weight(weight, tt, ll, nlayer_fit)
                else:
                    w, b = get_fitnet_weight(weight, 0, ll, nlayer_fit)
                    w = w * 0
                    b = b * 0
                # restrict the shift value of energy
                if (ll == (nlayer_fit - 1)):
                    b = b * 0
                bbi = self.wrap_bias(b, NBIT_SUM, NBIT_DATA_FL)
                bwi = self.wrap_weight(w, NBIT_WEIGHT, NBIT_WEIGHT_FL)
                bbt.append(bbi)
                bwt.append(bwi)
            bb.append(bbt)
            bw.append(bwt)
        #
        bfps, bbps = [], []
        for ss in range(NSEL):
            tt = ss // NSTDM
            sc = ss % NSTDM
            sr = ss % NSTDM
            bfp, bbp = '', ''
            for ll in range(nlayer_fit):
                nr = NNODE_FITS[ll]
                nc = NNODE_FITS[ll + 1]
                nrs = int(np.ceil(nr / NSTDM))
                ncs = int(np.ceil(nc / NSTDM))
                if (nc == 1):
                    # final layer
                    # fp #
                    bi = [bw[ll][tt][sr * nrs + rr][cc] for rr in range(nrs) for cc in range(nc)]
                    bi.reverse()
                    bfp = ''.join(bi) + bfp
                    #
                    bi = [bb[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                    bi.reverse()
                    bfp = ''.join(bi) + bfp
                    # bp #
                    bi = [bw[ll][tt][sr * nrs + rr][cc] for rr in range(nrs) for cc in range(nc)]
                    bi.reverse()
                    bbp = ''.join(bi) + bbp
                    #
                    bi = [bb[ll][tt][sc * ncs * 0 + cc] for cc in range(ncs)]
                    bi.reverse()
                    bbp = ''.join(bi) + bbp
                else:
                    # fp #
                    bi = [bw[ll][tt][rr][sc * ncs + cc] for cc in range(ncs) for rr in range(nr)]
                    bi.reverse()
                    bfp = ''.join(bi) + bfp
                    #
                    bi = [bb[ll][tt][sc * ncs + cc] for cc in range(ncs)]
                    bi.reverse()
                    bfp = ''.join(bi) + bfp
                    # bp #
                    bi = [bw[ll][tt][sr * nrs + rr][cc] for rr in range(nrs) for cc in range(nc)]
                    bi.reverse()
                    bbp = ''.join(bi) + bbp
                    #
                    bi = [bb[ll][tt][sc * ncs + cc] for cc in range(ncs)]
                    bi.reverse()
                    bbp = ''.join(bi) + bbp
            bfps.append(bfp)
            bbps.append(bbp)
        return bfps, bbps

    def wrap_bias(self, bias, NBIT_SUM, NBIT_DATA_FL):
        e = Encode()
        bias = e.qr(bias, NBIT_DATA_FL)
        Bs = e.dec2bin(bias, NBIT_SUM, True)
        return Bs

    def wrap_weight(self, weight, NBIT_WEIGHT, NBIT_WEIGHT_FL):
        sh = weight.shape
        nr, nc = sh[0], sh[1]
        e = Encode()
        weight = e.qr(weight, NBIT_WEIGHT_FL)
        Ws = e.dec2bin(weight, NBIT_WEIGHT, True)
        Ws = [[Ws[nc * rr + cc] for cc in range(nc)] for rr in range(nr)]
        return Ws

    def wrap_map(self):
        r"""Wrap the mapping table of embedding network
        """
        dscp = nvnmd_cfg.dscp
        maps = nvnmd_cfg.map
        nbit = nvnmd_cfg.nbit

        M1 = dscp['M1']
        ntype = dscp['ntype']
        NBIT_FEA = nbit['NBIT_FEA']
        NBIT_FEA_FL = nbit['NBIT_FEA_FL']

        keys = 's,sr,G'.split(',')
        keys2 = 'ds_dr2,dsr_dr2,dG_ds'.split(',')

        e = Encode()

        datas = {}
        datas2 = {}
        idxs = [[0, tt] for tt in range(ntype)]
        for ii in range(len(idxs)):
            tt, tt2 = idxs[ii]
            postfix = f'_t{tt}_t{tt2}'
            for key in (keys + keys2):
                if ii == 0:
                    datas[key] = []
                    datas2[key] = []
                datas[key].append(maps[key + postfix][0])  # v
                datas2[key].append(maps[key + postfix][1])  # dv

        for key in (keys + keys2):
            datas[key] = np.vstack(datas[key])
            datas[key] = e.qr(datas[key], NBIT_FEA_FL)

            datas2[key] = np.vstack(datas2[key])
            datas2[key] = e.qr(datas2[key], NBIT_FEA_FL)
        # fea
        dat = [datas[key] for key in keys] + [datas2[key] for key in keys]
        idx = np.int32(np.arange(0, int((M1 + 2) * 2)).reshape([2, -1]).transpose().reshape(-1))
        dat = np.hstack(dat)
        dat = dat[:, ::-1]
        dat = dat[:, idx]  # data consists of value and delta_value
        bs = e.dec2bin(dat, NBIT_FEA, True, 'fea')
        bs = e.merge_bin(bs, (M1 + 2) * 2)
        bfea = bs
        # gra
        dat = [datas[key] for key in keys2] + [datas2[key] for key in keys2]
        dat = np.hstack(dat)
        dat = dat[:, ::-1]
        dat = dat[:, idx]
        bs = e.dec2bin(dat, NBIT_FEA, True, 'gra')
        bs = e.merge_bin(bs, (M1 + 2) * 2)
        bgra = bs
        return bfea, bgra


def wrap(
    *,
    nvnmd_config: Optional[str] = 'nvnmd/config.npy',
    nvnmd_weight: Optional[str] = 'nvnmd/weight.npy',
    nvnmd_map: Optional[str] = 'nvnmd/map.npy',
    nvnmd_model: Optional[str] = 'nvnmd/model.pb',
    **kwargs
):
    wrapObj = Wrap(nvnmd_config, nvnmd_weight, nvnmd_map, nvnmd_model)
    wrapObj.wrap()
