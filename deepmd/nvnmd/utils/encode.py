
import numpy as np

from deepmd.nvnmd.utils.fio import FioHead 
from deepmd.nvnmd.data.data import jdata_sys


class Encode():

    def __init__(self) -> None:
        pass 
    
    def qr(self, v, nbit: int=14):
        return np.round(v * (2**nbit))

    def qf(self, v, nbit: int=14):
        return np.floor(v * (2**nbit))

    def qc(self, v, nbit: int=14):
        return np.ceil(v * (2**nbit))

    def check_dec(self, idec, nbit, signed=False, name=''):
        prec = np.int64(2**nbit)
        if signed:
            pmax = prec // 2 - 1
            pmin = -pmax 
        else:
            pmax = prec - 1
            pmin = 0
        I = idec < pmin 
        I2 = idec > pmax  

        if jdata_sys['debug']:
            if np.sum(I) > 0:
                head = FioHead.warning()
                print(f"{head}: there are data {name} smaller than the lower limit {pmin}")
            if np.sum(I2) > 0:
                head = FioHead.warning()
                print(f"{head}: there are data {name} bigger than the upper limit {pmax}")

    
    def extend_list(self, slbin, nfull):
        nfull = int(nfull)
        n = len(slbin)
        dn = nfull - n 
        ds = '0' * len(slbin[0])
        return slbin+[ds for ii in range(dn)]

    def extend_bin(self, slbin, nfull):
        nfull = int(nfull)
        n = len(slbin[0])
        dn = nfull - n 
        ds = '0' * int(dn) 
        return [ds+s for s in slbin]

    def extend_hex(self, slbin, nfull):
        nfull = int(nfull)
        n = len(slbin[0])
        dn = (nfull//4) - n 
        ds = '0' * int(dn) 
        return [ds+s for s in slbin]

    def split_bin(self, sbin, nbit: int):
        if (type(sbin) == list):
            sl = []
            for s in sbin:
                sl.extend(self.split_bin(s, nbit))
            return sl
        else:
            n = len(sbin)
            nseg = int(np.ceil(n / nbit))
            s = '0' * int(nseg*nbit - n)
            sbin = s + sbin 

            sl = [sbin[ii*nbit:(ii+1)*nbit] for ii in range(nseg)]
            sl = sl[::-1] 
            return sl 
    
    def reverse_bin(self, slbin, nreverse):
        nreverse = int(nreverse)
        # consider that {len(slbin)} can not be divided by {nreverse} without remainder
        n = int(np.ceil(len(slbin) / nreverse)) 
        slbin = self.extend_list(slbin, n * nreverse)
        return [slbin[ii*nreverse+nreverse-1-jj] for ii in range(n) for jj in range(nreverse)]
    
    def merge_bin(self, slbin, nmerge):
        nmerge = int(nmerge)
        # consider that {len(slbin)} can not be divided by {nmerge} without remainder
        n = int(np.ceil(len(slbin) / nmerge))
        slbin = self.extend_list(slbin, n * nmerge)
        return [''.join(slbin[nmerge*ii:nmerge*(ii+1)]) for ii in range(n)]

    def dec2bin(self, idec, nbit=10, signed=False, name=''):
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
        idec = idec + 2*prec 
        
        sl = []
        n = len(idec)
        for ii in range(n):
            s = bin(idec[ii])
            s = s[-nbit:] 
            sl.append(s) 
        
        return sl 
    
    def hex2bin_str(self, shex):
        n = len(shex)
        sl = []
        for ii in range(n):
            si = bin(int(shex[ii], 16)+16)
            sl.append(si[-4:])
        return ''.join(sl)

    def hex2bin(self, data):
        data = np.reshape(np.array(data), [-1])
        return [self.hex2bin_str(d) for d in data]
    
    def bin2hex_str(self, sbin):
        n = len(sbin)
        nx = int(np.ceil(n / 4))
        sbin = ('0'*(nx*4-n))+sbin
        sl = []
        for ii in range(nx):
            si = hex(int(sbin[4*ii:4*(ii+1)], 2)+16)
            sl.append(si[-1])
        return ''.join(sl)

    def bin2hex(self, data):
        data = np.reshape(np.array(data), [-1])
        return [self.bin2hex_str(d) for d in data]




# def test():
#     e = Encode()
#     dat = np.arange(-50, 50) / 20
#     bs = e.dec2bin(dat, 10, True)
#     hs = e.bin2hex(bs)

#     print(dat)
#     print(bs)
#     print(hs)



