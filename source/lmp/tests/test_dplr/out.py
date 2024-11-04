import numpy as np
expected_f_lr = np.array(
    [
        [0.20445234, 0.27936500, -0.23179282],
        [-0.30801828, -0.15412533, -0.17021364],
        [-0.44078300, 0.34719898, 0.63462716],
        [0.22103191, -0.50831649, -0.36328848],
        [0.60333935, -0.04531002, 0.15891833],
        [-0.28002232, 0.08118786, -0.02825056],
    ]
)

dist_metal2si = 1.0e-10
ener_metal2si = 1.3806504e-23 / 8.617343e-5
force_metal2si = ener_metal2si / dist_metal2si

print(expected_f_lr * force_metal2si)
