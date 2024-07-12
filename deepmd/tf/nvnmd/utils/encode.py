# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.tf.nvnmd.data.data import (
    jdata_sys,
)

log = logging.getLogger(__name__)


class Encode:
    r"""Encoding value as hex, bin, and dec format."""

    def __init__(self):
        pass

    def qr(self, v, nbit: int = 14):
        r"""Quantize value using round."""
        return np.round(v * (2**nbit))

    def qf(self, v, nbit: int = 14):
        r"""Quantize value using floor."""
        return np.floor(v * (2**nbit))

    def qc(self, v, nbit: int = 14):
        r"""Quantize value using ceil."""
        return np.ceil(v * (2**nbit))

    def split_expo_mant(self, v, min=-1000):
        vabs = np.abs(v)
        expo = np.log2(vabs)
        expo = np.maximum(expo, min)
        prec = 1.0 / 2.0**expo
        mant = v * prec
        return expo, mant

    def find_max_expo(self, v, expo_min=-1000):
        vabs = np.abs(v)
        vmax = np.max(vabs)
        expo_max = np.log2(vmax + 1e-50)
        expo_max = np.maximum(expo_max, expo_min)
        expo_max = np.floor(expo_max)
        return expo_max

    def norm_expo(self, v, nbit_frac=20, expo_min=-1000):
        expo_max = self.find_max_expo(v, expo_min)
        prec_expo = 2 ** (nbit_frac - expo_max)
        prec = 2**nbit_frac
        sign = np.sign(v)
        vabs = np.abs(v)
        vabs = np.floor(vabs * prec_expo) / prec
        return sign * vabs, expo_max

    def flt2bin_one(self, v, nbit_expo, nbit_frac):
        v = float(v)  # 64-bit float
        h = v.hex()
        n = len(h)
        st = n
        for ii in range(n):
            if h[ii] == "x":
                st = ii + 1
            if h[ii] == "p":
                ed = ii + 1
        is_zero = h[st] == "0"
        #
        if is_zero:
            return "0" * (1 + nbit_expo + nbit_frac)
        else:
            s = "1" if h[0] == "-" else "0"
            e = int(h[ed:]) + int(2 ** (nbit_expo - 1) + 2**nbit_expo)
            e = bin(e)[3:]  # 0b1xxxxxxx
            fh = h[st + 2 : ed - 1]
            fb = self.hex2bin_str(fh)
            f = fb[0:nbit_frac]
            return s + e + f

    def flt2bin(self, data, nbit_expo, nbit_frac):
        r"""Convert float into binary string list."""
        data = np.reshape(np.array(data), [-1])
        return [self.flt2bin_one(d, nbit_expo, nbit_frac) for d in data]

    def byte2hex(self, bs, nbyte):
        r"""Convert byte into hex
        bs: low byte in the first
        hex: low byte in the right.
        """
        nl = len(bs) // nbyte
        hs = []
        for ii in range(nl):
            b = bs[nbyte * ii : nbyte * (ii + 1)]
            b = b[::-1]
            h = b.hex()
            hs.append(h)
        return hs

    def check_dec(self, idec, nbit, signed=False, name=""):
        r"""Check whether the data (idec) is in the range
        range is :math:`[0, 2^nbit-1]` for unsigned
        range is :math:`[-2^{nbit-1}, 2^{nbit-1}-1]` for signed.
        """
        prec = np.int64(2**nbit)
        if signed:
            pmax = prec // 2 - 1
            pmin = -pmax
        else:
            pmax = prec - 1
            pmin = 0
        I1 = idec < pmin
        I2 = idec > pmax

        if jdata_sys["debug"]:
            if np.sum(I1) > 0:
                log.warning(
                    f"NVNMD: there are data {name} smaller than the lower limit {pmin}"
                )
            if np.sum(I2) > 0:
                log.warning(
                    f"NVNMD: there are data {name} bigger than the upper limit {pmax}"
                )

    def extend_list(self, slbin, nfull):
        r"""Extend the list (slbin) to the length (nfull)
        the attched element of list is 0.

        such as, when

        | slbin = ['10010','10100'],
        | nfull = 4

        extent it to

        ['10010','10100','00000','00000]
        """
        nfull = int(nfull)
        n = len(slbin)
        dn = nfull - n
        ds = "0" * len(slbin[0])
        return slbin + [ds for ii in range(dn)]

    def extend_bin(self, slbin, nfull):
        r"""Extend the element of list (slbin) to the length (nfull).

        such as, when

        | slbin = ['10010','10100'],
        | nfull = 6

        extent to

        ['010010','010100']
        """
        nfull = int(nfull)
        n = len(slbin[0])
        dn = nfull - n
        ds = "0" * int(dn)
        return [ds + s for s in slbin]

    def extend_hex(self, slhex, nfull):
        r"""Extend the element of list (slhex) to the length (nfull)."""
        nfull = int(nfull)
        n = len(slhex[0])
        dn = (nfull // 4) - n
        ds = "0" * int(dn)
        return [ds + s for s in slhex]

    def split_bin(self, sbin, nbit: int):
        r"""Split sbin into many segment with the length nbit."""
        if isinstance(sbin, list):
            sl = []
            for s in sbin:
                sl.extend(self.split_bin(s, nbit))
            return sl
        else:
            n = len(sbin)
            nseg = int(np.ceil(n / nbit))
            s = "0" * int(nseg * nbit - n)
            sbin = s + sbin

            sl = [sbin[ii * nbit : (ii + 1) * nbit] for ii in range(nseg)]
            sl = sl[::-1]
            return sl

    def reverse_bin(self, slbin, nreverse):
        r"""Reverse binary string list per `nreverse` value."""
        nreverse = int(nreverse)
        # consider that {len(slbin)} can not be divided by {nreverse} without remainder
        n = int(np.ceil(len(slbin) / nreverse))
        slbin = self.extend_list(slbin, n * nreverse)
        return [
            slbin[ii * nreverse + nreverse - 1 - jj]
            for ii in range(n)
            for jj in range(nreverse)
        ]

    def merge_bin(self, slbin, nmerge):
        r"""Merge binary string list per `nmerge` value."""
        nmerge = int(nmerge)
        # consider that {len(slbin)} can not be divided by {nmerge} without remainder
        n = int(np.ceil(len(slbin) / nmerge))
        slbin = self.extend_list(slbin, n * nmerge)
        return ["".join(slbin[nmerge * ii : nmerge * (ii + 1)]) for ii in range(n)]

    def dec2bin(self, idec, nbit=10, signed=False, name=""):
        r"""Convert dec array to binary string list."""
        idec = np.int64(np.reshape(np.array(idec), [-1]))
        self.check_dec(idec, nbit, signed, name)

        prec = np.int64(2**nbit)
        if signed:
            pmax = prec // 2 - 1
            pmin = -pmax
        else:
            pmax = prec - 1
            pmin = 0
        idec = np.maximum(pmin, idec)
        idec = np.minimum(pmax, idec)
        idec = idec + 2 * prec

        sl = []
        n = len(idec)
        for ii in range(n):
            s = bin(idec[ii])
            s = s[-nbit:]
            sl.append(s)
        return sl

    def hex2bin_str(self, shex):
        r"""Convert hex string to binary string."""
        n = len(shex)
        sl = []
        for ii in range(n):
            si = bin(int(shex[ii], 16) + 16)
            sl.append(si[-4:])
        return "".join(sl)

    def hex2bin(self, data):
        r"""Convert hex string list to binary string list."""
        data = np.reshape(np.array(data), [-1])
        return [self.hex2bin_str(d) for d in data]

    def bin2hex_str(self, sbin):
        r"""Convert binary string to hex string."""
        n = len(sbin)
        nx = int(np.ceil(n / 4))
        sbin = ("0" * (nx * 4 - n)) + sbin
        sl = []
        for ii in range(nx):
            si = hex(int(sbin[4 * ii : 4 * (ii + 1)], 2) + 16)
            sl.append(si[-1])
        return "".join(sl)

    def bin2hex(self, data):
        r"""Convert binary string list to hex string list."""
        data = np.reshape(np.array(data), [-1])
        return [self.bin2hex_str(d) for d in data]
