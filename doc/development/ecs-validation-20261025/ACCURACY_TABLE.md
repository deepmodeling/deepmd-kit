# ECS 多模型多尺寸精度测试

> 精度优先；未放宽阈值。误差范围为每对输出max_abs的min～max，不是置信区间。基线SHA与模型SHA见JSON。

模式：TORCH=纯PyTorch编译；ECS=当前PR候选（非Neo为回退）；ECS_RUNTIME=强制真实8192分批。单位：能量/位力eV，力eV/Å，stress eV/Å³。所有结果均不是DFT预测误差。

| 模型 | 原子数 | 对照 | 范围 | 完成进程数 | 能量max_abs范围 | 力max_abs范围 | 位力max_abs范围 | stress max_abs范围 | 最差误差/容差 | 状态 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| Mini | 96 | TORCH vs TORCH | within_process | 3/3 | 1.788e-07～1.234e-05 | 4.585e-06～8.618e-06 | 1.402e-05～6.538e-05 | 2.375e-08～1.200e-07 | 1.1463 | FAIL |
| Mini | 96 | TORCH vs TORCH | across_process | 3/3 | 0.000e+00～1.246e-05 | 4.587e-06～8.061e-06 | 7.968e-06～6.371e-05 | 1.463e-08～1.170e-07 | 1.5990 | FAIL |
| Mini | 96 | BASE vs BASE | within_process | 3/3 | 1.192e-07～9.894e-06 | 4.567e-06～6.735e-06 | 1.251e-05～4.619e-05 | 2.294e-08～8.481e-08 | 1.8636 | FAIL |
| Mini | 96 | BASE vs BASE | across_process | 3/3 | 5.960e-08～1.049e-05 | 4.451e-06～7.551e-06 | 1.017e-05～4.644e-05 | 1.809e-08～8.526e-08 | 1.7011 | FAIL |
| Mini | 96 | ECS vs ECS | within_process | 3/3 | 3.576e-07～1.740e-05 | 4.543e-06～7.322e-06 | 7.101e-06～7.174e-05 | 1.285e-08～1.317e-07 | 2.1289 | FAIL |
| Mini | 96 | ECS vs ECS | across_process | 3/3 | 5.960e-08～1.699e-05 | 4.675e-06～8.280e-06 | 8.842e-06～5.695e-05 | 1.572e-08～1.046e-07 | 2.1372 | FAIL |
| Mini | 96 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～1.836e-05 | 5.616e-06～1.080e-05 | 1.972e-05～8.237e-05 | 3.388e-08～1.512e-07 | 1.7538 | FAIL |
| Mini | 96 | ECS vs TORCH | cross_implementation | 3/3 | 0.000e+00～1.651e-05 | 5.894e-06～1.135e-05 | 1.419e-05～9.370e-05 | 2.501e-08～1.720e-07 | 1.9425 | FAIL |
| Mini | 96 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～1.663e-05 | 4.214e-06～8.067e-06 | 8.232e-06～8.385e-05 | 1.474e-08～1.540e-07 | 1.9484 | FAIL |
| Mini | 512 | TORCH vs TORCH | within_process | 3/3 | 4.172e-07～3.606e-05 | 5.870e-06～7.873e-06 | 2.876e-05～1.708e-04 | 9.493e-09～5.882e-08 | 2.6615 | FAIL |
| Mini | 512 | TORCH vs TORCH | across_process | 3/3 | 1.788e-07～3.493e-05 | 5.543e-06～8.601e-06 | 2.627e-05～1.340e-04 | 9.046e-09～4.613e-08 | 2.1807 | FAIL |
| Mini | 512 | BASE vs BASE | within_process | 3/3 | 7.153e-07～3.594e-05 | 5.392e-06～8.266e-06 | 2.302e-05～9.672e-05 | 7.924e-09～3.330e-08 | 2.4707 | FAIL |
| Mini | 512 | BASE vs BASE | across_process | 3/3 | 1.192e-07～2.855e-05 | 5.616e-06～8.582e-06 | 1.416e-05～1.244e-04 | 4.757e-09～4.283e-08 | 2.6165 | FAIL |
| Mini | 512 | ECS vs ECS | within_process | 3/3 | 5.960e-07～2.813e-05 | 5.659e-06～9.532e-06 | 1.953e-05～1.400e-04 | 6.450e-09～4.821e-08 | 1.9644 | FAIL |
| Mini | 512 | ECS vs ECS | across_process | 3/3 | 0.000e+00～2.646e-05 | 5.619e-06～8.691e-06 | 1.610e-05～1.445e-04 | 5.544e-09～4.975e-08 | 3.1341 | FAIL |
| Mini | 512 | BASE vs TORCH | cross_implementation | 3/3 | 5.960e-08～3.952e-05 | 7.673e-06～1.210e-05 | 9.353e-05～2.595e-04 | 3.220e-08～8.933e-08 | 2.7111 | FAIL |
| Mini | 512 | ECS vs TORCH | cross_implementation | 3/3 | 5.960e-08～3.040e-05 | 7.604e-06～1.289e-05 | 9.723e-05～2.485e-04 | 3.347e-08～8.555e-08 | 3.0424 | FAIL |
| Mini | 512 | ECS vs BASE | cross_implementation | 3/3 | 5.960e-08～3.386e-05 | 5.100e-06～8.801e-06 | 1.883e-05～1.636e-04 | 6.483e-09～5.634e-08 | 3.0773 | FAIL |
| Mini | 4096 | TORCH vs TORCH | within_process | 3/3 | 5.364e-07～7.576e-05 | 6.887e-06～1.004e-05 | 9.010e-05～3.499e-04 | 3.878e-09～1.506e-08 | 10.2623 | FAIL |
| Mini | 4096 | TORCH vs TORCH | across_process | 3/3 | 1.788e-07～7.707e-05 | 6.973e-06～8.997e-06 | 6.730e-05～3.137e-04 | 2.584e-09～1.350e-08 | 8.9322 | FAIL |
| Mini | 4096 | BASE vs BASE | within_process | 3/3 | 2.086e-06～9.507e-05 | 6.874e-06～9.211e-06 | 8.355e-05～2.777e-04 | 3.595e-09～1.195e-08 | 10.3628 | FAIL |
| Mini | 4096 | BASE vs BASE | across_process | 3/3 | 7.749e-07～1.156e-04 | 7.001e-06～9.328e-06 | 5.222e-05～3.119e-04 | 2.040e-09～1.342e-08 | 12.0549 | FAIL |
| Mini | 4096 | ECS vs ECS | within_process | 3/3 | 5.960e-07～7.117e-05 | 7.187e-06～1.007e-05 | 5.141e-05～3.646e-04 | 2.061e-09～1.569e-08 | 6.1755 | FAIL |
| Mini | 4096 | ECS vs ECS | across_process | 3/3 | 5.364e-07～8.416e-05 | 7.002e-06～9.466e-06 | 4.396e-05～3.296e-04 | 1.892e-09～1.418e-08 | 7.3596 | FAIL |
| Mini | 4096 | BASE vs TORCH | cross_implementation | 3/3 | 3.576e-07～1.388e-04 | 9.301e-06～1.306e-05 | 7.796e-04～1.488e-03 | 3.355e-08～6.403e-08 | 12.5994 | FAIL |
| Mini | 4096 | ECS vs TORCH | cross_implementation | 3/3 | 1.311e-06～1.465e-04 | 9.302e-06～1.398e-05 | 8.275e-04～1.521e-03 | 3.561e-08～6.544e-08 | 9.8370 | FAIL |
| Mini | 4096 | ECS vs BASE | cross_implementation | 3/3 | 2.384e-07～1.084e-04 | 6.848e-06～1.005e-05 | 3.919e-05～3.468e-04 | 1.687e-09～1.492e-08 | 11.9793 | FAIL |
| Nano | 96 | TORCH vs TORCH | within_process | 3/3 | 4.768e-07～2.658e-05 | 7.035e-06～1.130e-05 | 1.325e-05～7.839e-05 | 2.318e-08～1.439e-07 | 5.5420 | FAIL |
| Nano | 96 | TORCH vs TORCH | across_process | 3/3 | 0.000e+00～2.420e-05 | 6.704e-06～1.136e-05 | 1.439e-05～7.676e-05 | 2.480e-08～1.409e-07 | 5.4077 | FAIL |
| Nano | 96 | BASE vs BASE | within_process | 3/3 | 1.192e-07～1.299e-05 | 6.957e-06～1.220e-05 | 1.406e-05～7.483e-05 | 2.405e-08～1.374e-07 | 5.7063 | FAIL |
| Nano | 96 | BASE vs BASE | across_process | 3/3 | 1.788e-07～1.878e-05 | 6.726e-06～1.142e-05 | 1.593e-05～8.811e-05 | 2.581e-08～1.618e-07 | 8.3241 | FAIL |
| Nano | 96 | ECS vs ECS | within_process | 3/3 | 0.000e+00～1.794e-05 | 7.266e-06～1.080e-05 | 1.871e-05～8.601e-05 | 3.432e-08～1.579e-07 | 7.3608 | FAIL |
| Nano | 96 | ECS vs ECS | across_process | 3/3 | 1.192e-07～2.122e-05 | 7.020e-06～1.150e-05 | 1.177e-05～8.074e-05 | 2.161e-08～1.482e-07 | 6.4631 | FAIL |
| Nano | 96 | BASE vs TORCH | cross_implementation | 3/3 | 1.788e-07～3.499e-05 | 6.454e-06～1.286e-05 | 1.219e-05～9.567e-05 | 2.155e-08～1.757e-07 | 9.4133 | FAIL |
| Nano | 96 | ECS vs TORCH | cross_implementation | 3/3 | 0.000e+00～3.982e-05 | 6.870e-06～1.269e-05 | 1.227e-05～9.163e-05 | 2.207e-08～1.682e-07 | 8.0976 | FAIL |
| Nano | 96 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～2.354e-05 | 6.351e-06～1.155e-05 | 1.137e-05～1.014e-04 | 2.003e-08～1.862e-07 | 9.1350 | FAIL |
| Nano | 512 | TORCH vs TORCH | within_process | 3/3 | 8.941e-07～3.463e-05 | 8.984e-06～1.488e-05 | 4.949e-05～1.978e-04 | 1.683e-08～6.809e-08 | 7.7673 | FAIL |
| Nano | 512 | TORCH vs TORCH | across_process | 3/3 | 3.576e-07～6.020e-05 | 8.737e-06～1.247e-05 | 3.414e-05～2.010e-04 | 1.174e-08～6.921e-08 | 7.8147 | FAIL |
| Nano | 512 | BASE vs BASE | within_process | 3/3 | 1.371e-06～5.305e-05 | 8.899e-06～1.278e-05 | 4.218e-05～1.863e-04 | 1.452e-08～6.415e-08 | 4.2656 | FAIL |
| Nano | 512 | BASE vs BASE | across_process | 3/3 | 8.941e-07～6.390e-05 | 8.700e-06～1.347e-05 | 1.671e-05～1.607e-04 | 5.273e-09～5.534e-08 | 4.6439 | FAIL |
| Nano | 512 | ECS vs ECS | within_process | 3/3 | 1.907e-06～4.596e-05 | 8.712e-06～1.348e-05 | 3.527e-05～1.795e-04 | 1.214e-08～6.181e-08 | 7.0513 | FAIL |
| Nano | 512 | ECS vs ECS | across_process | 3/3 | 1.073e-06～5.358e-05 | 8.399e-06～1.288e-05 | 1.549e-05～2.203e-04 | 5.095e-09～7.585e-08 | 7.1989 | FAIL |
| Nano | 512 | BASE vs TORCH | cross_implementation | 3/3 | 2.980e-07～1.107e-04 | 9.036e-06～1.477e-05 | 2.200e-05～2.575e-04 | 6.814e-09～8.866e-08 | 7.1940 | FAIL |
| Nano | 512 | ECS vs TORCH | cross_implementation | 3/3 | 6.676e-06～1.102e-04 | 8.284e-06～1.325e-05 | 2.255e-05～2.560e-04 | 7.763e-09～8.813e-08 | 9.4950 | FAIL |
| Nano | 512 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～6.342e-05 | 8.381e-06～1.570e-05 | 2.904e-05～1.981e-04 | 9.659e-09～6.821e-08 | 6.2237 | FAIL |
| Nano | 4096 | TORCH vs TORCH | within_process | 3/3 | 1.848e-06～1.260e-04 | 1.119e-05～1.436e-05 | 5.544e-05～3.779e-04 | 2.386e-09～1.626e-08 | 2.4958 | FAIL |
| Nano | 4096 | TORCH vs TORCH | across_process | 3/3 | 7.749e-07～1.612e-04 | 1.085e-05～1.484e-05 | 7.050e-05～4.504e-04 | 3.034e-09～1.938e-08 | 2.6872 | FAIL |
| Nano | 4096 | BASE vs BASE | within_process | 3/3 | 1.091e-05～1.924e-04 | 1.079e-05～1.563e-05 | 8.116e-05～4.467e-04 | 3.240e-09～1.922e-08 | 2.2079 | FAIL |
| Nano | 4096 | BASE vs BASE | across_process | 3/3 | 7.749e-07～1.666e-04 | 1.064e-05～1.480e-05 | 9.755e-05～5.898e-04 | 4.198e-09～2.538e-08 | 3.8540 | FAIL |
| Nano | 4096 | ECS vs ECS | within_process | 3/3 | 2.801e-06～1.946e-04 | 1.118e-05～1.424e-05 | 9.154e-05～4.406e-04 | 3.940e-09～1.896e-08 | 2.3048 | FAIL |
| Nano | 4096 | ECS vs ECS | across_process | 3/3 | 1.132e-06～2.133e-04 | 1.075e-05～1.605e-05 | 6.212e-05～5.332e-04 | 2.673e-09～2.294e-08 | 4.1555 | FAIL |
| Nano | 4096 | BASE vs TORCH | cross_implementation | 3/3 | 1.556e-04～4.887e-04 | 1.069e-05～1.613e-05 | 5.668e-05～7.157e-04 | 2.388e-09～3.080e-08 | 4.1304 | FAIL |
| Nano | 4096 | ECS vs TORCH | cross_implementation | 3/3 | 1.753e-04～5.112e-04 | 1.066e-05～1.554e-05 | 7.483e-05～7.943e-04 | 3.220e-09～3.418e-08 | 4.4355 | FAIL |
| Nano | 4096 | ECS vs BASE | cross_implementation | 3/3 | 2.384e-07～2.006e-04 | 1.020e-05～1.522e-05 | 3.754e-05～6.358e-04 | 1.501e-09～2.736e-08 | 4.1590 | FAIL |
| Neo | 96 | TORCH vs TORCH | within_process | 3/3 | 0.000e+00～1.222e-05 | 2.529e-06～4.930e-06 | 4.051e-06～2.011e-05 | 6.975e-09～3.692e-08 | 0.5965 | PASS |
| Neo | 96 | TORCH vs TORCH | across_process | 3/3 | 1.788e-07～1.091e-05 | 2.529e-06～4.594e-06 | 4.727e-06～2.222e-05 | 8.078e-09～4.080e-08 | 0.4535 | PASS |
| Neo | 96 | BASE vs BASE | within_process | 3/3 | 4.768e-07～7.212e-06 | 2.483e-06～3.981e-06 | 3.664e-06～2.443e-05 | 4.932e-09～4.486e-08 | 0.3532 | PASS |
| Neo | 96 | BASE vs BASE | across_process | 3/3 | 5.960e-08～1.073e-05 | 2.437e-06～4.474e-06 | 3.807e-06～1.991e-05 | 6.350e-09～3.656e-08 | 0.4794 | PASS |
| Neo | 96 | ECS vs ECS | within_process | 3/3 | 5.960e-08～6.557e-06 | 2.617e-06～4.197e-06 | 2.986e-06～2.181e-05 | 5.483e-09～4.005e-08 | 0.4760 | PASS |
| Neo | 96 | ECS vs ECS | across_process | 3/3 | 1.192e-07～6.855e-06 | 2.545e-06～4.369e-06 | 3.881e-06～2.690e-05 | 7.126e-09～4.938e-08 | 0.5175 | PASS |
| Neo | 96 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～1.115e-05 | 3.603e-06～7.196e-06 | 1.558e-05～4.824e-05 | 2.862e-08～8.858e-08 | 0.5819 | PASS |
| Neo | 96 | ECS vs TORCH | cross_implementation | 3/3 | 0.000e+00～1.127e-05 | 3.614e-06～7.185e-06 | 1.331e-05～5.232e-05 | 2.154e-08～9.607e-08 | 0.5914 | PASS |
| Neo | 96 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～1.037e-05 | 2.304e-06～5.034e-06 | 2.399e-06～2.851e-05 | 2.107e-09～5.235e-08 | 0.5769 | PASS |
| Neo | 512 | TORCH vs TORCH | within_process | 3/3 | 1.192e-07～1.776e-05 | 3.427e-06～4.336e-06 | 1.278e-05～5.254e-05 | 4.298e-09～1.809e-08 | 0.4674 | PASS |
| Neo | 512 | TORCH vs TORCH | across_process | 3/3 | 4.172e-07～1.734e-05 | 3.283e-06～4.901e-06 | 1.086e-05～6.951e-05 | 3.545e-09～2.393e-08 | 0.7340 | PASS |
| Neo | 512 | BASE vs BASE | within_process | 3/3 | 1.192e-07～2.247e-05 | 3.262e-06～4.286e-06 | 7.840e-06～6.508e-05 | 2.699e-09～2.240e-08 | 0.6034 | PASS |
| Neo | 512 | BASE vs BASE | across_process | 3/3 | 5.960e-08～1.889e-05 | 2.996e-06～5.144e-06 | 8.897e-06～6.434e-05 | 2.589e-09～2.215e-08 | 0.5044 | PASS |
| Neo | 512 | ECS vs ECS | within_process | 3/3 | 9.537e-07～1.884e-05 | 3.293e-06～5.012e-06 | 9.156e-06～5.708e-05 | 3.152e-09～1.965e-08 | 0.8473 | PASS |
| Neo | 512 | ECS vs ECS | across_process | 3/3 | 1.192e-07～2.056e-05 | 3.137e-06～4.858e-06 | 6.668e-06～5.049e-05 | 2.046e-09～1.738e-08 | 0.8394 | PASS |
| Neo | 512 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～2.021e-05 | 4.529e-06～7.150e-06 | 7.708e-05～1.673e-04 | 2.654e-08～5.758e-08 | 0.8254 | PASS |
| Neo | 512 | ECS vs TORCH | cross_implementation | 3/3 | 2.980e-07～2.718e-05 | 4.555e-06～7.020e-06 | 7.569e-05～1.572e-04 | 2.606e-08～5.411e-08 | 1.0809 | FAIL |
| Neo | 512 | ECS vs BASE | cross_implementation | 3/3 | 1.192e-07～2.670e-05 | 3.075e-06～5.255e-06 | 4.760e-06～7.743e-05 | 1.186e-09～2.666e-08 | 0.8589 | PASS |
| Neo | 4096 | BASE vs BASE | within_process | 3/3 | 1.788e-07～5.084e-05 | 3.915e-06～5.114e-06 | 2.331e-05～1.037e-04 | 1.003e-09～4.284e-09 | 1.3472 | FAIL |
| Neo | 4096 | BASE vs BASE | across_process | 3/3 | 5.364e-07～5.960e-05 | 4.095e-06～5.564e-06 | 2.725e-05～1.374e-04 | 1.066e-09～5.913e-09 | 1.7886 | FAIL |
| Neo | 4096 | ECS vs ECS | within_process | 3/3 | 2.086e-06～5.770e-05 | 4.087e-06～5.089e-06 | 3.967e-05～2.099e-04 | 1.707e-09～9.031e-09 | 2.6929 | FAIL |
| Neo | 4096 | ECS vs ECS | across_process | 3/3 | 1.788e-07～5.788e-05 | 4.010e-06～5.939e-06 | 2.293e-05～2.537e-04 | 8.575e-10～1.092e-08 | 3.4631 | FAIL |
| Neo | 4096 | ECS vs BASE | cross_implementation | 3/3 | 1.788e-07～6.694e-05 | 4.074e-06～6.208e-06 | 1.792e-05～1.883e-04 | 6.963e-10～8.102e-09 | 2.7953 | FAIL |
| NeoCu | 96 | TORCH vs TORCH | within_process | 3/3 | 1.192e-07～3.338e-06 | 3.603e-07～5.453e-07 | 1.724e-06～1.361e-05 | 1.519e-09～1.200e-08 | 0.1450 | PASS |
| NeoCu | 96 | TORCH vs TORCH | across_process | 3/3 | 0.000e+00～3.636e-06 | 3.280e-07～6.306e-07 | 1.289e-06～1.229e-05 | 1.134e-09～1.083e-08 | 0.1805 | PASS |
| NeoCu | 96 | BASE vs BASE | within_process | 3/3 | 5.960e-08～4.292e-06 | 3.674e-07～6.496e-07 | 1.058e-06～1.315e-05 | 9.327e-10～1.159e-08 | 0.2355 | PASS |
| NeoCu | 96 | BASE vs BASE | across_process | 3/3 | 0.000e+00～4.590e-06 | 3.296e-07～5.808e-07 | 1.086e-06～1.663e-05 | 9.277e-10～1.466e-08 | 0.2466 | PASS |
| NeoCu | 96 | ECS vs ECS | within_process | 3/3 | 1.192e-07～4.292e-06 | 3.467e-07～5.431e-07 | 1.491e-06～1.204e-05 | 1.314e-09～1.061e-08 | 0.2134 | PASS |
| NeoCu | 96 | ECS vs ECS | across_process | 3/3 | 0.000e+00～4.411e-06 | 3.581e-07～6.476e-07 | 9.126e-07～1.739e-05 | 8.043e-10～1.533e-08 | 0.2576 | PASS |
| NeoCu | 96 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～6.855e-06 | 3.360e-07～6.391e-07 | 3.217e-06～2.707e-05 | 2.835e-09～2.386e-08 | 0.2037 | PASS |
| NeoCu | 96 | ECS vs TORCH | cross_implementation | 3/3 | 0.000e+00～5.782e-06 | 3.374e-07～6.297e-07 | 3.805e-06～2.837e-05 | 3.353e-09～2.500e-08 | 0.2452 | PASS |
| NeoCu | 96 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～5.841e-06 | 3.036e-07～6.730e-07 | 5.077e-07～1.968e-05 | 4.474e-10～1.734e-08 | 0.2689 | PASS |
| NeoCu | 512 | TORCH vs TORCH | within_process | 3/3 | 0.000e+00～1.115e-05 | 4.107e-07～5.963e-07 | 3.552e-06～3.151e-05 | 5.869e-10～5.207e-09 | 0.2895 | PASS |
| NeoCu | 512 | TORCH vs TORCH | across_process | 3/3 | 3.576e-07～1.138e-05 | 4.138e-07～6.573e-07 | 2.813e-06～4.521e-05 | 4.174e-10～7.471e-09 | 0.2794 | PASS |
| NeoCu | 512 | BASE vs BASE | within_process | 3/3 | 1.192e-07～8.345e-06 | 4.176e-07～5.775e-07 | 4.065e-06～3.114e-05 | 6.577e-10～5.145e-09 | 0.3275 | PASS |
| NeoCu | 512 | BASE vs BASE | across_process | 3/3 | 0.000e+00～1.007e-05 | 4.184e-07～6.110e-07 | 1.907e-06～3.137e-05 | 3.124e-10～5.184e-09 | 0.3759 | PASS |
| NeoCu | 512 | ECS vs ECS | within_process | 3/3 | 6.557e-07～6.557e-06 | 4.320e-07～7.186e-07 | 2.182e-06～4.652e-05 | 3.606e-10～7.687e-09 | 0.2868 | PASS |
| NeoCu | 512 | ECS vs ECS | across_process | 3/3 | 5.960e-08～1.085e-05 | 4.208e-07～6.428e-07 | 2.016e-06～4.702e-05 | 3.332e-10～7.770e-09 | 0.4818 | PASS |
| NeoCu | 512 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～1.919e-05 | 4.175e-07～7.357e-07 | 4.216e-05～1.130e-04 | 6.966e-09～1.867e-08 | 0.3659 | PASS |
| NeoCu | 512 | ECS vs TORCH | cross_implementation | 3/3 | 5.960e-08～1.860e-05 | 4.282e-07～7.158e-07 | 3.795e-05～1.295e-04 | 6.271e-09～2.139e-08 | 0.3641 | PASS |
| NeoCu | 512 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～1.073e-05 | 3.929e-07～6.560e-07 | 1.895e-06～4.981e-05 | 2.801e-10～8.231e-09 | 0.4547 | PASS |
| NeoCu | 4096 | TORCH vs TORCH | within_process | 3/3 | 8.345e-07～1.687e-05 | 5.199e-07～6.523e-07 | 7.247e-06～8.839e-05 | 1.497e-10～1.826e-09 | 0.3163 | PASS |
| NeoCu | 4096 | TORCH vs TORCH | across_process | 3/3 | 5.960e-08～2.599e-05 | 5.038e-07～7.152e-07 | 8.798e-06～1.036e-04 | 1.817e-10～2.140e-09 | 0.3102 | PASS |
| NeoCu | 4096 | BASE vs BASE | within_process | 3/3 | 7.153e-07～4.148e-05 | 5.221e-07～7.069e-07 | 5.660e-06～7.616e-05 | 9.913e-11～1.573e-09 | 0.2691 | PASS |
| NeoCu | 4096 | BASE vs BASE | across_process | 3/3 | 1.192e-07～3.791e-05 | 5.172e-07～6.890e-07 | 8.123e-06～1.054e-04 | 1.678e-10～2.178e-09 | 0.3652 | PASS |
| NeoCu | 4096 | ECS vs ECS | within_process | 3/3 | 1.788e-07～2.778e-05 | 5.439e-07～7.169e-07 | 7.803e-06～1.111e-04 | 1.612e-10～2.295e-09 | 0.2754 | PASS |
| NeoCu | 4096 | ECS vs ECS | across_process | 3/3 | 5.960e-08～3.821e-05 | 5.166e-07～7.105e-07 | 7.015e-06～9.401e-05 | 1.449e-10～1.942e-09 | 0.4257 | PASS |
| NeoCu | 4096 | BASE vs TORCH | cross_implementation | 3/3 | 3.046e-05～8.726e-05 | 5.351e-07～8.014e-07 | 4.939e-04～6.932e-04 | 1.020e-08～1.432e-08 | 0.4284 | PASS |
| NeoCu | 4096 | ECS vs TORCH | cross_implementation | 3/3 | 2.074e-05～8.494e-05 | 5.446e-07～7.932e-07 | 4.744e-04～6.950e-04 | 9.798e-09～1.435e-08 | 0.4357 | PASS |
| NeoCu | 4096 | ECS vs BASE | cross_implementation | 3/3 | 5.960e-08～3.868e-05 | 5.229e-07～7.972e-07 | 5.578e-06～1.249e-04 | 9.600e-11～2.581e-09 | 0.3727 | PASS |
| Pro | 96 | TORCH vs TORCH | within_process | 3/3 | 2.980e-08～3.546e-06 | 7.446e-07～1.197e-06 | 1.965e-06～9.613e-06 | 2.423e-09～1.765e-08 | 0.1584 | PASS |
| Pro | 96 | TORCH vs TORCH | across_process | 3/3 | 5.960e-08～4.083e-06 | 7.445e-07～1.667e-06 | 1.298e-06～8.369e-06 | 1.832e-09～1.537e-08 | 0.1908 | PASS |
| Pro | 96 | BASE vs BASE | within_process | 3/3 | 2.980e-08～2.086e-06 | 6.661e-07～1.003e-06 | 1.850e-06～6.390e-06 | 2.572e-09～1.173e-08 | 0.1034 | PASS |
| Pro | 96 | BASE vs BASE | across_process | 3/3 | 2.980e-08～3.368e-06 | 6.060e-07～1.153e-06 | 1.505e-06～8.763e-06 | 2.631e-09～1.609e-08 | 0.1715 | PASS |
| Pro | 96 | ECS vs ECS | within_process | 3/3 | 2.980e-08～2.652e-06 | 7.281e-07～1.056e-06 | 1.903e-06～7.248e-06 | 1.676e-09～1.331e-08 | 0.1771 | PASS |
| Pro | 96 | ECS vs ECS | across_process | 3/3 | 0.000e+00～3.338e-06 | 6.510e-07～1.144e-06 | 1.396e-06～7.585e-06 | 1.931e-09～1.393e-08 | 0.2516 | PASS |
| Pro | 96 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～5.037e-06 | 1.030e-06～1.919e-06 | 6.303e-06～1.613e-05 | 1.023e-08～2.962e-08 | 0.1988 | PASS |
| Pro | 96 | ECS vs TORCH | cross_implementation | 3/3 | 0.000e+00～5.871e-06 | 1.039e-06～1.691e-06 | 5.839e-06～1.643e-05 | 9.968e-09～3.017e-08 | 0.2784 | PASS |
| Pro | 96 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～3.725e-06 | 6.266e-07～1.310e-06 | 1.210e-06～9.066e-06 | 1.710e-09～1.665e-08 | 0.2276 | PASS |
| Pro | 512 | BASE vs BASE | within_process | 3/3 | 1.192e-07～7.242e-06 | 8.882e-07～1.212e-06 | 3.637e-06～1.652e-05 | 6.287e-10～5.687e-09 | 0.2266 | PASS |
| Pro | 512 | BASE vs BASE | across_process | 3/3 | 2.980e-08～1.019e-05 | 8.338e-07～1.360e-06 | 3.104e-06～2.018e-05 | 8.452e-10～6.947e-09 | 0.2604 | PASS |
| Pro | 512 | ECS vs ECS | within_process | 3/3 | 2.980e-08～7.421e-06 | 8.405e-07～1.356e-06 | 3.452e-06～1.421e-05 | 1.188e-09～4.893e-09 | 0.2375 | PASS |
| Pro | 512 | ECS vs ECS | across_process | 3/3 | 2.980e-08～8.345e-06 | 8.582e-07～1.497e-06 | 2.475e-06～1.641e-05 | 6.944e-10～5.648e-09 | 0.2399 | PASS |
| Pro | 512 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～1.121e-05 | 8.272e-07～1.477e-06 | 2.360e-06～1.753e-05 | 5.223e-10～6.034e-09 | 0.2884 | PASS |

## 所有进程状态（含OOM/失败，不能略去）

| 运行 | 状态 | 参数量 | opaque ECS调用 | single路径标记 |
| --- | --- | ---: | ---: | --- |
| Mini_n4096_BASE_p0 | completed | 654988 | 0 | False |
| Mini_n4096_BASE_p1 | completed | 654988 | 0 | False |
| Mini_n4096_BASE_p2 | completed | 654988 | 0 | False |
| Mini_n4096_ECS_p0 | completed | 654988 | 0 | False |
| Mini_n4096_ECS_p1 | completed | 654988 | 0 | False |
| Mini_n4096_ECS_p2 | completed | 654988 | 0 | False |
| Mini_n4096_TORCH_p0 | completed | 654988 | 0 | False |
| Mini_n4096_TORCH_p1 | completed | 654988 | 0 | False |
| Mini_n4096_TORCH_p2 | completed | 654988 | 0 | False |
| Mini_n512_BASE_p0 | completed | 654988 | 0 | False |
| Mini_n512_BASE_p1 | completed | 654988 | 0 | False |
| Mini_n512_BASE_p2 | completed | 654988 | 0 | False |
| Mini_n512_ECS_p0 | completed | 654988 | 0 | False |
| Mini_n512_ECS_p1 | completed | 654988 | 0 | False |
| Mini_n512_ECS_p2 | completed | 654988 | 0 | False |
| Mini_n512_TORCH_p0 | completed | 654988 | 0 | False |
| Mini_n512_TORCH_p1 | completed | 654988 | 0 | False |
| Mini_n512_TORCH_p2 | completed | 654988 | 0 | False |
| Mini_n96_BASE_p0 | completed | 654988 | 0 | False |
| Mini_n96_BASE_p1 | completed | 654988 | 0 | False |
| Mini_n96_BASE_p2 | completed | 654988 | 0 | False |
| Mini_n96_ECS_p0 | completed | 654988 | 0 | False |
| Mini_n96_ECS_p1 | completed | 654988 | 0 | False |
| Mini_n96_ECS_p2 | completed | 654988 | 0 | False |
| Mini_n96_TORCH_p0 | completed | 654988 | 0 | False |
| Mini_n96_TORCH_p1 | completed | 654988 | 0 | False |
| Mini_n96_TORCH_p2 | completed | 654988 | 0 | False |
| Nano_n4096_BASE_p0 | completed | 480268 | 0 | False |
| Nano_n4096_BASE_p1 | completed | 480268 | 0 | False |
| Nano_n4096_BASE_p2 | completed | 480268 | 0 | False |
| Nano_n4096_ECS_p0 | completed | 480268 | 0 | False |
| Nano_n4096_ECS_p1 | completed | 480268 | 0 | False |
| Nano_n4096_ECS_p2 | completed | 480268 | 0 | False |
| Nano_n4096_TORCH_p0 | completed | 480268 | 0 | False |
| Nano_n4096_TORCH_p1 | completed | 480268 | 0 | False |
| Nano_n4096_TORCH_p2 | completed | 480268 | 0 | False |
| Nano_n512_BASE_p0 | completed | 480268 | 0 | False |
| Nano_n512_BASE_p1 | completed | 480268 | 0 | False |
| Nano_n512_BASE_p2 | completed | 480268 | 0 | False |
| Nano_n512_ECS_p0 | completed | 480268 | 0 | False |
| Nano_n512_ECS_p1 | completed | 480268 | 0 | False |
| Nano_n512_ECS_p2 | completed | 480268 | 0 | False |
| Nano_n512_TORCH_p0 | completed | 480268 | 0 | False |
| Nano_n512_TORCH_p1 | completed | 480268 | 0 | False |
| Nano_n512_TORCH_p2 | completed | 480268 | 0 | False |
| Nano_n96_BASE_p0 | completed | 480268 | 0 | False |
| Nano_n96_BASE_p1 | completed | 480268 | 0 | False |
| Nano_n96_BASE_p2 | completed | 480268 | 0 | False |
| Nano_n96_ECS_p0 | completed | 480268 | 0 | False |
| Nano_n96_ECS_p1 | completed | 480268 | 0 | False |
| Nano_n96_ECS_p2 | completed | 480268 | 0 | False |
| Nano_n96_TORCH_p0 | completed | 480268 | 0 | False |
| Nano_n96_TORCH_p1 | completed | 480268 | 0 | False |
| Nano_n96_TORCH_p2 | completed | 480268 | 0 | False |
| NeoCu_n4096_BASE_p0 | completed | 1124910 | 0 | False |
| NeoCu_n4096_BASE_p1 | completed | 1124910 | 0 | False |
| NeoCu_n4096_BASE_p2 | completed | 1124910 | 0 | False |
| NeoCu_n4096_ECS_p0 | completed | 1124910 | 0 | True |
| NeoCu_n4096_ECS_p1 | completed | 1124910 | 0 | True |
| NeoCu_n4096_ECS_p2 | completed | 1124910 | 0 | True |
| NeoCu_n4096_TORCH_p0 | completed | 1124910 | 0 | False |
| NeoCu_n4096_TORCH_p1 | completed | 1124910 | 0 | False |
| NeoCu_n4096_TORCH_p2 | completed | 1124910 | 0 | False |
| NeoCu_n512_BASE_p0 | completed | 1124910 | 0 | False |
| NeoCu_n512_BASE_p1 | completed | 1124910 | 0 | False |
| NeoCu_n512_BASE_p2 | completed | 1124910 | 0 | False |
| NeoCu_n512_ECS_p0 | completed | 1124910 | 0 | True |
| NeoCu_n512_ECS_p1 | completed | 1124910 | 0 | True |
| NeoCu_n512_ECS_p2 | completed | 1124910 | 0 | True |
| NeoCu_n512_TORCH_p0 | completed | 1124910 | 0 | False |
| NeoCu_n512_TORCH_p1 | completed | 1124910 | 0 | False |
| NeoCu_n512_TORCH_p2 | completed | 1124910 | 0 | False |
| NeoCu_n96_BASE_p0 | completed | 1124910 | 0 | False |
| NeoCu_n96_BASE_p1 | completed | 1124910 | 0 | False |
| NeoCu_n96_BASE_p2 | completed | 1124910 | 0 | False |
| NeoCu_n96_ECS_p0 | completed | 1124910 | 0 | True |
| NeoCu_n96_ECS_p1 | completed | 1124910 | 0 | True |
| NeoCu_n96_ECS_p2 | completed | 1124910 | 0 | True |
| NeoCu_n96_TORCH_p0 | completed | 1124910 | 0 | False |
| NeoCu_n96_TORCH_p1 | completed | 1124910 | 0 | False |
| NeoCu_n96_TORCH_p2 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n4096_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n4096_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n4096_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n4096_TORCH_p0 | oom | 1124910 | 0 | False |
| Neo_n4096_TORCH_p1 | oom | 1124910 | 0 | False |
| Neo_n4096_TORCH_p2 | oom | 1124910 | 0 | False |
| Neo_n512_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n512_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n512_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n512_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n512_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n512_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n512_TORCH_p0 | completed | 1124910 | 0 | False |
| Neo_n512_TORCH_p1 | completed | 1124910 | 0 | False |
| Neo_n512_TORCH_p2 | completed | 1124910 | 0 | False |
| Neo_n96_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n96_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n96_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n96_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n96_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n96_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n96_TORCH_p0 | completed | 1124910 | 0 | False |
| Neo_n96_TORCH_p1 | completed | 1124910 | 0 | False |
| Neo_n96_TORCH_p2 | completed | 1124910 | 0 | False |
| Pro_n4096_BASE_p0 | oom | 22678990 | 0 | False |
| Pro_n4096_BASE_p1 | oom | 22678990 | 0 | False |
| Pro_n4096_BASE_p2 | oom | 22678990 | 0 | False |
| Pro_n4096_ECS_p0 | oom | 22678990 | 0 | False |
| Pro_n4096_ECS_p1 | oom | 22678990 | 0 | False |
| Pro_n4096_ECS_p2 | oom | 22678990 | 0 | False |
| Pro_n4096_TORCH_p0 | oom | 22678990 | 0 | False |
| Pro_n4096_TORCH_p1 | oom | 22678990 | 0 | False |
| Pro_n4096_TORCH_p2 | oom | 22678990 | 0 | False |
| Pro_n512_BASE_p0 | completed | 22678990 | 0 | False |
| Pro_n512_BASE_p1 | completed | 22678990 | 0 | False |
| Pro_n512_BASE_p2 | completed | 22678990 | 0 | False |
| Pro_n512_ECS_p0 | completed | 22678990 | 0 | False |
| Pro_n512_ECS_p1 | completed | 22678990 | 0 | False |
| Pro_n512_ECS_p2 | completed | 22678990 | 0 | False |
| Pro_n512_TORCH_p0 | oom | 22678990 | 0 | False |
| Pro_n512_TORCH_p1 | oom | 22678990 | 0 | False |
| Pro_n512_TORCH_p2 | oom | 22678990 | 0 | False |
| Pro_n96_BASE_p0 | completed | 22678990 | 0 | False |
| Pro_n96_BASE_p1 | completed | 22678990 | 0 | False |
| Pro_n96_BASE_p2 | completed | 22678990 | 0 | False |
| Pro_n96_ECS_p0 | completed | 22678990 | 0 | False |
| Pro_n96_ECS_p1 | completed | 22678990 | 0 | False |
| Pro_n96_ECS_p2 | completed | 22678990 | 0 | False |
| Pro_n96_TORCH_p0 | completed | 22678990 | 0 | False |
| Pro_n96_TORCH_p1 | completed | 22678990 | 0 | False |
| Pro_n96_TORCH_p2 | completed | 22678990 | 0 | False |
