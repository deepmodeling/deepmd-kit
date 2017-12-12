#!/usr/bin/env python3

import numpy as np

b = np.load ('box.npy')
if b.shape[1] == 9 : 
    print ("converted, do nothing")
    exit 
elif b.shape[1] != 6 :
    print ("wrong box format")
    exit

b0 = b[:,1] - b[:,0]
b1 = b[:,3] - b[:,2]
b2 = b[:,5] - b[:,4]

nb = np.zeros ((b.shape[0], 9))

nb[:,0] = b0
nb[:,4] = b1
nb[:,8] = b2

np.save ('box.npy', nb)


