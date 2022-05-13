

import numpy as np
from numpy.linalg import *
from ase.io import read


def store(coord, uc_std):
        while coord > uc_std:
                coord = coord - uc_std
        while coord < 0:
                coord = coord + uc_std
        return coord


def restore(posi, cell):
	inv_cell = inv(cell)
	coord = np.dot(posi, inv_cell)
	#np.savetxt('coord', coord, fmt="%.9f")
	[lx, ly] = coord.shape

	kk = np.zeros((lx, ly))
	for i in range(lx):
		for j in range(ly):
			kk[i][j] = store(coord[i][j], 1)
			#print(kk)

#	np.savetxt('direct-coord', kk, fmt="%.9f")

	newposi=np.dot(kk, cell)
#	np.savetxt('cart-coord', newposi, fmt="%.9f")
	return newposi







box = np.loadtxt('box.raw')
coord = np.loadtxt('coord.raw')
print(coord.shape)
[ lx, ly ] = coord.shape
na = int(ly / 3)
#print(na)

tar_coord = np.zeros((lx, ly))

for i in range(lx):
	cell = np.reshape(box[i], (3, 3))
	posi = np.reshape(coord[i], (na, 3))
	newposi = restore(posi, cell)		
	tar_coord[i] = newposi.flatten()


#print(tar_coord.shape)

np.savetxt('newcoord.raw', tar_coord, fmt="%.8f")





