# ECS 多模型多尺寸精度测试

> 精度优先；未放宽阈值。误差范围为每对输出max_abs的min～max，不是置信区间。基线SHA与模型SHA见JSON。

模式：TORCH=纯PyTorch编译；ECS=当前PR候选（非Neo为回退）；ECS_RUNTIME=强制真实8192分批。单位：能量/位力eV，力eV/Å，stress eV/Å³。所有结果均不是DFT预测误差。

| 模型 | 原子数 | 对照 | 范围 | 完成进程数 | 能量max_abs范围 | 力max_abs范围 | 位力max_abs范围 | stress max_abs范围 | 最差误差/容差 | 状态 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| Mini | 512 | TORCH vs TORCH | within_process | 3/3 | 1.788e-07～1.907e-05 | 5.743e-06～7.458e-06 | 1.941e-05～1.008e-04 | 6.477e-09～3.453e-08 | 2.3256 | FAIL |
| Mini | 512 | TORCH vs TORCH | across_process | 3/3 | 0.000e+00～2.682e-05 | 5.574e-06～8.402e-06 | 2.161e-05～1.017e-04 | 7.439e-09～3.501e-08 | 2.0800 | FAIL |
| Mini | 512 | BASE vs BASE | within_process | 3/3 | 8.941e-07～2.909e-05 | 5.665e-06～7.634e-06 | 1.822e-05～8.902e-05 | 5.847e-09～3.065e-08 | 1.3209 | FAIL |
| Mini | 512 | BASE vs BASE | across_process | 3/3 | 2.384e-07～3.576e-05 | 5.394e-06～8.533e-06 | 1.780e-05～9.780e-05 | 6.128e-09～3.367e-08 | 1.5409 | FAIL |
| Mini | 512 | ECS vs ECS | within_process | 3/3 | 2.384e-07～2.784e-05 | 5.507e-06～8.241e-06 | 2.388e-05～1.102e-04 | 7.637e-09～3.793e-08 | 2.3481 | FAIL |
| Mini | 512 | ECS vs ECS | across_process | 3/3 | 2.980e-07～2.706e-05 | 5.538e-06～8.991e-06 | 8.570e-06～1.185e-04 | 2.600e-09～4.078e-08 | 2.7696 | FAIL |
| Mini | 512 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～3.856e-05 | 7.663e-06～1.194e-05 | 8.734e-05～2.392e-04 | 3.007e-08～8.237e-08 | 2.0564 | FAIL |
| Mini | 512 | ECS vs TORCH | cross_implementation | 3/3 | 1.192e-07～3.630e-05 | 7.538e-06～1.362e-05 | 8.123e-05～2.064e-04 | 2.797e-08～7.105e-08 | 2.6359 | FAIL |
| Mini | 512 | ECS vs BASE | cross_implementation | 3/3 | 1.192e-07～3.225e-05 | 5.317e-06～8.227e-06 | 7.091e-06～1.170e-04 | 2.441e-09～4.027e-08 | 2.1640 | FAIL |
| Mini | 1000 | TORCH vs TORCH | within_process | 3/3 | 7.749e-07～3.952e-05 | 5.942e-06～8.479e-06 | 3.820e-05～1.486e-04 | 6.734e-09～2.619e-08 | 0.6828 | PASS |
| Mini | 1000 | TORCH vs TORCH | across_process | 3/3 | 4.172e-07～5.251e-05 | 6.128e-06～8.859e-06 | 2.881e-05～1.862e-04 | 4.861e-09～3.283e-08 | 0.7997 | PASS |
| Mini | 1000 | BASE vs BASE | within_process | 3/3 | 5.364e-07～3.475e-05 | 5.791e-06～8.084e-06 | 2.814e-05～9.627e-05 | 4.864e-09～1.697e-08 | 0.7113 | PASS |
| Mini | 1000 | BASE vs BASE | across_process | 3/3 | 1.788e-07～5.239e-05 | 5.877e-06～9.181e-06 | 1.580e-05～1.545e-04 | 2.721e-09～2.723e-08 | 0.7256 | PASS |
| Mini | 1000 | ECS vs ECS | within_process | 3/3 | 2.384e-07～3.195e-05 | 5.835e-06～8.294e-06 | 3.425e-05～1.184e-04 | 6.019e-09～2.087e-08 | 1.1129 | FAIL |
| Mini | 1000 | ECS vs ECS | across_process | 3/3 | 2.384e-07～4.518e-05 | 5.955e-06～8.708e-06 | 1.679e-05～1.293e-04 | 2.959e-09～2.279e-08 | 1.3280 | FAIL |
| Mini | 1000 | BASE vs TORCH | cross_implementation | 3/3 | 1.192e-07～6.008e-05 | 8.291e-06～1.272e-05 | 1.898e-04～4.217e-04 | 3.346e-08～7.434e-08 | 0.9772 | PASS |
| Mini | 1000 | ECS vs TORCH | cross_implementation | 3/3 | 5.960e-08～4.542e-05 | 7.863e-06～1.166e-05 | 2.127e-04～3.895e-04 | 3.749e-08～6.865e-08 | 1.0280 | FAIL |
| Mini | 1000 | ECS vs BASE | cross_implementation | 3/3 | 2.384e-07～5.299e-05 | 5.502e-06～9.310e-06 | 1.849e-05～1.536e-04 | 3.259e-09～2.707e-08 | 1.2858 | FAIL |
| Mini | 4096 | TORCH vs TORCH | within_process | 3/3 | 5.960e-08～6.229e-05 | 7.276e-06～9.493e-06 | 5.062e-05～4.385e-04 | 2.066e-09～1.887e-08 | 12.0785 | FAIL |
| Mini | 4096 | TORCH vs TORCH | across_process | 3/3 | 5.960e-08～8.333e-05 | 6.791e-06～9.902e-06 | 5.727e-05～3.834e-04 | 2.084e-09～1.650e-08 | 18.0056 | FAIL |
| Mini | 4096 | BASE vs BASE | within_process | 3/3 | 3.099e-06～1.399e-04 | 7.088e-06～8.594e-06 | 4.909e-05～2.918e-04 | 2.043e-09～1.256e-08 | 7.5328 | FAIL |
| Mini | 4096 | BASE vs BASE | across_process | 3/3 | 3.576e-07～1.013e-04 | 6.853e-06～9.191e-06 | 4.477e-05～3.680e-04 | 1.894e-09～1.584e-08 | 8.7236 | FAIL |
| Mini | 4096 | ECS vs ECS | within_process | 3/3 | 8.941e-07～6.741e-05 | 6.751e-06～9.570e-06 | 8.410e-05～3.134e-04 | 3.504e-09～1.342e-08 | 7.6971 | FAIL |
| Mini | 4096 | ECS vs ECS | across_process | 3/3 | 2.980e-07～8.285e-05 | 6.717e-06～1.040e-05 | 3.166e-05～3.210e-04 | 1.362e-09～1.381e-08 | 8.5980 | FAIL |
| Mini | 4096 | BASE vs TORCH | cross_implementation | 3/3 | 1.073e-06～1.630e-04 | 9.423e-06～1.370e-05 | 8.078e-04～1.487e-03 | 3.476e-08～6.401e-08 | 16.3663 | FAIL |
| Mini | 4096 | ECS vs TORCH | cross_implementation | 3/3 | 1.609e-06～1.435e-04 | 9.178e-06～1.420e-05 | 7.984e-04～1.493e-03 | 3.436e-08～6.425e-08 | 14.4638 | FAIL |
| Mini | 4096 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～1.230e-04 | 6.839e-06～9.656e-06 | 4.190e-05～3.580e-04 | 1.803e-09～1.541e-08 | 10.4884 | FAIL |
| Nano | 512 | TORCH vs TORCH | within_process | 3/3 | 1.788e-07～5.186e-05 | 8.767e-06～1.729e-05 | 4.055e-05～1.745e-04 | 1.365e-08～6.008e-08 | 5.1961 | FAIL |
| Nano | 512 | TORCH vs TORCH | across_process | 3/3 | 8.941e-07～5.031e-05 | 8.257e-06～1.488e-05 | 3.397e-05～1.774e-04 | 1.067e-08～6.108e-08 | 8.1770 | FAIL |
| Nano | 512 | BASE vs BASE | within_process | 3/3 | 6.557e-07～5.233e-05 | 8.955e-06～1.241e-05 | 4.078e-05～1.411e-04 | 1.404e-08～4.859e-08 | 4.3052 | FAIL |
| Nano | 512 | BASE vs BASE | across_process | 3/3 | 3.576e-07～4.852e-05 | 8.638e-06～1.356e-05 | 2.392e-05～1.993e-04 | 7.193e-09～6.862e-08 | 5.5042 | FAIL |
| Nano | 512 | ECS vs ECS | within_process | 3/3 | 2.384e-07～4.870e-05 | 8.850e-06～1.174e-05 | 2.709e-05～1.370e-04 | 9.077e-09～4.717e-08 | 3.9393 | FAIL |
| Nano | 512 | ECS vs ECS | across_process | 3/3 | 2.980e-07～6.038e-05 | 8.533e-06～1.260e-05 | 2.392e-05～2.191e-04 | 8.234e-09～7.544e-08 | 5.7037 | FAIL |
| Nano | 512 | BASE vs TORCH | cross_implementation | 3/3 | 2.980e-06～1.152e-04 | 8.867e-06～1.386e-05 | 1.153e-05～2.624e-04 | 3.971e-09～9.033e-08 | 9.4910 | FAIL |
| Nano | 512 | ECS vs TORCH | cross_implementation | 3/3 | 4.172e-07～1.128e-04 | 8.495e-06～1.357e-05 | 2.651e-05～2.647e-04 | 9.126e-09～9.113e-08 | 8.6337 | FAIL |
| Nano | 512 | ECS vs BASE | cross_implementation | 3/3 | 4.172e-07～5.412e-05 | 8.595e-06～1.269e-05 | 2.358e-05～2.068e-04 | 8.119e-09～7.121e-08 | 6.5564 | FAIL |
| Nano | 1000 | TORCH vs TORCH | within_process | 3/3 | 1.788e-07～6.706e-05 | 9.814e-06～1.328e-05 | 5.537e-05～2.621e-04 | 9.760e-09～4.621e-08 | 5.8245 | FAIL |
| Nano | 1000 | TORCH vs TORCH | across_process | 3/3 | 8.941e-07～7.910e-05 | 9.717e-06～1.356e-05 | 3.931e-05～2.863e-04 | 6.794e-09～5.047e-08 | 7.5700 | FAIL |
| Nano | 1000 | BASE vs BASE | within_process | 3/3 | 1.073e-06～5.555e-05 | 8.786e-06～1.246e-05 | 4.376e-05～2.458e-04 | 7.638e-09～4.333e-08 | 6.4322 | FAIL |
| Nano | 1000 | BASE vs BASE | across_process | 3/3 | 5.960e-08～7.153e-05 | 9.121e-06～1.302e-05 | 5.281e-05～2.499e-04 | 9.309e-09～4.405e-08 | 9.2287 | FAIL |
| Nano | 1000 | ECS vs ECS | within_process | 3/3 | 1.192e-06～5.585e-05 | 9.348e-06～1.242e-05 | 4.847e-05～2.198e-04 | 8.545e-09～3.874e-08 | 4.9302 | FAIL |
| Nano | 1000 | ECS vs ECS | across_process | 3/3 | 5.960e-08～6.944e-05 | 9.239e-06～1.430e-05 | 2.857e-05～2.319e-04 | 5.036e-09～4.087e-08 | 6.9685 | FAIL |
| Nano | 1000 | BASE vs TORCH | cross_implementation | 3/3 | 1.049e-05～1.621e-04 | 9.189e-06～1.489e-05 | 3.178e-05～2.637e-04 | 5.601e-09～4.649e-08 | 9.2328 | FAIL |
| Nano | 1000 | ECS vs TORCH | cross_implementation | 3/3 | 1.264e-05～1.663e-04 | 9.170e-06～1.386e-05 | 1.395e-05～3.074e-04 | 2.459e-09～5.418e-08 | 8.0526 | FAIL |
| Nano | 1000 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～7.570e-05 | 9.041e-06～1.358e-05 | 3.918e-05～2.690e-04 | 6.906e-09～4.742e-08 | 8.1419 | FAIL |
| Nano | 4096 | TORCH vs TORCH | within_process | 3/3 | 3.755e-06～1.802e-04 | 1.138e-05～1.498e-05 | 9.571e-05～6.979e-04 | 4.119e-09～3.003e-08 | 2.3384 | FAIL |
| Nano | 4096 | TORCH vs TORCH | across_process | 3/3 | 5.960e-08～2.024e-04 | 1.065e-05～1.443e-05 | 9.505e-05～6.595e-04 | 3.965e-09～2.838e-08 | 2.8159 | FAIL |
| Nano | 4096 | BASE vs BASE | within_process | 3/3 | 2.384e-06～1.985e-04 | 1.098e-05～1.587e-05 | 1.133e-04～3.956e-04 | 4.860e-09～1.703e-08 | 1.3307 | FAIL |
| Nano | 4096 | BASE vs BASE | across_process | 3/3 | 1.907e-06～2.370e-04 | 1.060e-05～1.419e-05 | 5.914e-05～4.544e-04 | 2.379e-09～1.956e-08 | 2.5569 | FAIL |
| Nano | 4096 | ECS vs ECS | within_process | 3/3 | 3.278e-06～1.050e-04 | 1.101e-05～1.347e-05 | 6.502e-05～7.514e-04 | 2.450e-09～3.233e-08 | 1.2806 | FAIL |
| Nano | 4096 | ECS vs ECS | across_process | 3/3 | 1.788e-07～1.045e-04 | 1.095e-05～1.462e-05 | 8.139e-05～6.386e-04 | 3.503e-09～2.748e-08 | 2.1274 | FAIL |
| Nano | 4096 | BASE vs TORCH | cross_implementation | 3/3 | 9.382e-05～4.325e-04 | 1.070e-05～1.538e-05 | 8.534e-05～1.047e-03 | 3.673e-09～4.504e-08 | 3.3957 | FAIL |
| Nano | 4096 | ECS vs TORCH | cross_implementation | 3/3 | 1.401e-04～4.290e-04 | 1.060e-05～1.602e-05 | 9.420e-05～1.060e-03 | 3.617e-09～4.560e-08 | 3.0524 | FAIL |
| Nano | 4096 | ECS vs BASE | cross_implementation | 3/3 | 5.960e-08～1.978e-04 | 1.056e-05～1.622e-05 | 7.473e-05～6.500e-04 | 3.042e-09～2.797e-08 | 2.4384 | FAIL |
| Neo | 512 | TORCH vs TORCH | within_process | 3/3 | 5.960e-08～1.943e-05 | 3.261e-06～4.747e-06 | 1.320e-05～7.500e-05 | 4.517e-09～2.582e-08 | 0.6232 | PASS |
| Neo | 512 | TORCH vs TORCH | across_process | 3/3 | 4.768e-07～1.884e-05 | 3.108e-06～5.441e-06 | 9.732e-06～7.461e-05 | 2.689e-09～2.569e-08 | 0.6851 | PASS |
| Neo | 512 | BASE vs BASE | within_process | 3/3 | 5.960e-08～2.521e-05 | 3.148e-06～4.934e-06 | 1.098e-05～4.944e-05 | 3.581e-09～1.702e-08 | 0.6621 | PASS |
| Neo | 512 | BASE vs BASE | across_process | 3/3 | 2.384e-07～2.354e-05 | 3.135e-06～5.108e-06 | 9.663e-06～6.807e-05 | 3.327e-09～2.344e-08 | 0.7136 | PASS |
| Neo | 512 | ECS vs ECS | within_process | 3/3 | 1.192e-07～1.627e-05 | 3.263e-06～4.824e-06 | 5.221e-06～4.653e-05 | 1.513e-09～1.602e-08 | 0.7407 | PASS |
| Neo | 512 | ECS vs ECS | across_process | 3/3 | 0.000e+00～1.740e-05 | 3.247e-06～4.922e-06 | 1.290e-05～6.690e-05 | 3.928e-09～2.303e-08 | 0.8633 | PASS |
| Neo | 512 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～3.117e-05 | 4.321e-06～6.981e-06 | 7.577e-05～1.771e-04 | 2.609e-08～6.098e-08 | 1.1932 | FAIL |
| Neo | 512 | ECS vs TORCH | cross_implementation | 3/3 | 1.192e-07～2.927e-05 | 4.556e-06～7.410e-06 | 7.720e-05～1.594e-04 | 2.658e-08～5.487e-08 | 1.0295 | FAIL |
| Neo | 512 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～2.331e-05 | 3.149e-06～5.301e-06 | 7.728e-06～7.278e-05 | 2.660e-09～2.506e-08 | 0.9591 | PASS |
| Neo | 512 | ECS_RUNTIME vs ECS_RUNTIME | within_process | 3/3 | 4.172e-07～1.824e-05 | 3.403e-06～4.479e-06 | 1.187e-05～4.710e-05 | 2.963e-09～1.622e-08 | 1.0518 | FAIL |
| Neo | 512 | ECS_RUNTIME vs ECS_RUNTIME | across_process | 3/3 | 5.960e-08～1.431e-05 | 3.251e-06～5.502e-06 | 6.900e-06～5.424e-05 | 2.375e-09～1.867e-08 | 0.9080 | PASS |
| Neo | 512 | ECS_RUNTIME vs BASE | cross_implementation | 3/3 | 5.960e-08～2.301e-05 | 3.042e-06～5.382e-06 | 9.281e-06～6.285e-05 | 2.715e-09～2.164e-08 | 1.0467 | FAIL |
| Neo | 512 | ECS_RUNTIME vs TORCH | cross_implementation | 3/3 | 5.960e-08～2.372e-05 | 4.383e-06～7.113e-06 | 7.607e-05～1.648e-04 | 2.619e-08～5.675e-08 | 1.0368 | FAIL |
| Neo | 512 | ECS_RUNTIME vs ECS | cross_implementation | 3/3 | 0.000e+00～2.110e-05 | 3.024e-06～5.139e-06 | 6.926e-06～6.315e-05 | 2.384e-09～2.174e-08 | 1.1043 | FAIL |
| Neo | 1000 | TORCH vs TORCH | within_process | 3/3 | 5.960e-07～2.939e-05 | 3.633e-06～5.042e-06 | 1.796e-05～8.294e-05 | 2.496e-09～1.462e-08 | 0.4762 | PASS |
| Neo | 1000 | TORCH vs TORCH | across_process | 3/3 | 1.192e-07～2.736e-05 | 3.471e-06～5.588e-06 | 1.483e-05～1.008e-04 | 2.181e-09～1.777e-08 | 0.4978 | PASS |
| Neo | 1000 | BASE vs BASE | within_process | 3/3 | 7.153e-07～2.074e-05 | 3.335e-06～4.864e-06 | 1.662e-05～7.371e-05 | 2.929e-09～1.299e-08 | 0.3498 | PASS |
| Neo | 1000 | BASE vs BASE | across_process | 3/3 | 5.960e-08～1.848e-05 | 3.363e-06～5.305e-06 | 1.574e-05～1.023e-04 | 2.277e-09～1.802e-08 | 0.5508 | PASS |
| Neo | 1000 | ECS vs ECS | within_process | 3/3 | 6.557e-07～3.302e-05 | 3.544e-06～4.635e-06 | 1.599e-05～7.633e-05 | 2.645e-09～1.346e-08 | 0.3679 | PASS |
| Neo | 1000 | ECS vs ECS | across_process | 3/3 | 2.980e-07～3.064e-05 | 3.362e-06～5.389e-06 | 1.043e-05～1.018e-04 | 1.839e-09～1.794e-08 | 0.5019 | PASS |
| Neo | 1000 | BASE vs TORCH | cross_implementation | 3/3 | 0.000e+00～2.736e-05 | 4.921e-06～7.257e-06 | 1.523e-04～2.917e-04 | 2.685e-08～5.141e-08 | 0.6108 | PASS |
| Neo | 1000 | ECS vs TORCH | cross_implementation | 3/3 | 1.192e-07～4.029e-05 | 5.026e-06～8.691e-06 | 1.513e-04～3.357e-04 | 2.667e-08～5.917e-08 | 0.5704 | PASS |
| Neo | 1000 | ECS vs BASE | cross_implementation | 3/3 | 0.000e+00～3.189e-05 | 3.352e-06～5.042e-06 | 1.021e-05～1.058e-04 | 1.644e-09～1.866e-08 | 0.5937 | PASS |
| Neo | 1000 | ECS_RUNTIME vs ECS_RUNTIME | within_process | 3/3 | 2.384e-07～2.861e-05 | 3.570e-06～4.866e-06 | 1.601e-05～1.015e-04 | 2.815e-09～1.790e-08 | 0.4164 | PASS |
| Neo | 1000 | ECS_RUNTIME vs ECS_RUNTIME | across_process | 3/3 | 2.384e-07～2.778e-05 | 3.505e-06～5.059e-06 | 1.429e-05～1.104e-04 | 2.186e-09～1.946e-08 | 0.3538 | PASS |
| Neo | 1000 | ECS_RUNTIME vs BASE | cross_implementation | 3/3 | 5.960e-08～2.718e-05 | 3.289e-06～4.839e-06 | 1.072e-05～1.038e-04 | 1.889e-09～1.829e-08 | 0.6028 | PASS |
| Neo | 1000 | ECS_RUNTIME vs TORCH | cross_implementation | 3/3 | 5.960e-08～3.272e-05 | 4.816e-06～7.409e-06 | 1.576e-04～3.508e-04 | 2.778e-08～6.184e-08 | 0.5441 | PASS |
| Neo | 1000 | ECS_RUNTIME vs ECS | cross_implementation | 3/3 | 5.960e-08～3.481e-05 | 3.344e-06～5.309e-06 | 1.161e-05～1.169e-04 | 1.612e-09～2.061e-08 | 0.5110 | PASS |
| Neo | 4096 | BASE vs BASE | within_process | 3/3 | 2.384e-07～4.643e-05 | 4.007e-06～5.565e-06 | 4.326e-05～1.764e-04 | 1.428e-09～7.592e-09 | 2.0354 | FAIL |
| Neo | 4096 | BASE vs BASE | across_process | 3/3 | 4.172e-07～5.627e-05 | 3.912e-06～5.939e-06 | 2.275e-05～2.513e-04 | 9.788e-10～1.081e-08 | 2.8536 | FAIL |
| Neo | 4096 | ECS vs ECS | within_process | 3/3 | 2.384e-07～5.406e-05 | 4.204e-06～5.183e-06 | 2.174e-05～1.084e-04 | 5.853e-10～4.666e-09 | 2.2326 | FAIL |
| Neo | 4096 | ECS vs ECS | across_process | 3/3 | 3.576e-07～8.285e-05 | 4.080e-06～5.679e-06 | 2.126e-05～1.705e-04 | 5.108e-10～7.337e-09 | 2.8543 | FAIL |
| Neo | 4096 | ECS vs BASE | cross_implementation | 3/3 | 5.960e-08～6.813e-05 | 4.013e-06～5.806e-06 | 2.113e-05～2.482e-04 | 6.362e-10～1.068e-08 | 3.2394 | FAIL |
| Neo | 4096 | ECS_RUNTIME vs ECS_RUNTIME | within_process | 3/3 | 3.576e-07～5.656e-05 | 4.011e-06～5.388e-06 | 2.321e-05～2.127e-04 | 9.989e-10～9.155e-09 | 3.0024 | FAIL |
| Neo | 4096 | ECS_RUNTIME vs ECS_RUNTIME | across_process | 3/3 | 2.384e-07～6.253e-05 | 4.014e-06～6.203e-06 | 2.701e-05～1.597e-04 | 9.890e-10～6.871e-09 | 2.9657 | FAIL |
| Neo | 4096 | ECS_RUNTIME vs BASE | cross_implementation | 3/3 | 5.960e-08～6.562e-05 | 4.056e-06～6.068e-06 | 1.560e-05～2.561e-04 | 6.714e-10～1.102e-08 | 3.5269 | FAIL |
| Neo | 4096 | ECS_RUNTIME vs ECS | cross_implementation | 3/3 | 5.960e-08～7.308e-05 | 3.972e-06～5.791e-06 | 2.029e-05～1.955e-04 | 8.732e-10～8.411e-09 | 3.1677 | FAIL |
| Pro | 64 | TORCH vs TORCH | within_process | 1/1 | 5.960e-08～1.520e-06 | 7.994e-07～1.084e-06 | 1.829e-06～4.709e-06 | 3.323e-09～1.297e-08 | 0.0749 | PASS（样本未齐） |
| Pro | 64 | BASE vs BASE | within_process | 1/1 | 2.086e-07～2.235e-06 | 7.070e-07～9.719e-07 | 1.706e-06～4.243e-06 | 4.698e-09～1.169e-08 | 0.0727 | PASS（样本未齐） |
| Pro | 64 | ECS vs ECS | within_process | 2/2 | 2.980e-08～3.010e-06 | 6.763e-07～9.147e-07 | 2.130e-06～7.250e-06 | 3.415e-09～1.997e-08 | 0.0893 | PASS（样本未齐） |
| Pro | 64 | ECS vs ECS | across_process | 2/2 | 2.980e-08～4.113e-06 | 6.775e-07～1.166e-06 | 1.785e-06～6.365e-06 | 2.901e-09～1.753e-08 | 0.1206 | PASS（样本未齐） |
| Pro | 64 | BASE vs TORCH | cross_implementation | 1/1 | 1.788e-07～3.934e-06 | 1.086e-06～1.643e-06 | 2.214e-06～9.813e-06 | 5.613e-09～2.703e-08 | 0.0992 | PASS（样本未齐） |
| Pro | 64 | ECS vs TORCH | cross_implementation | 2/1 | 2.086e-07～4.351e-06 | 9.742e-07～1.560e-06 | 2.394e-06～1.112e-05 | 5.669e-09～3.062e-08 | 0.1440 | PASS（样本未齐） |
| Pro | 64 | ECS vs BASE | cross_implementation | 2/1 | 5.960e-08～3.636e-06 | 6.614e-07～1.147e-06 | 1.603e-06～6.327e-06 | 2.958e-09～1.743e-08 | 0.0945 | PASS（样本未齐） |

## 所有进程状态（含OOM/失败，不能略去）

| 运行 | 状态 | 参数量 | opaque ECS调用 | single路径标记 |
| --- | --- | ---: | ---: | --- |
| Mini_n1000_BASE_p0 | completed | 654988 | 0 | False |
| Mini_n1000_BASE_p1 | completed | 654988 | 0 | False |
| Mini_n1000_BASE_p2 | completed | 654988 | 0 | False |
| Mini_n1000_ECS_p0 | completed | 654988 | 0 | False |
| Mini_n1000_ECS_p1 | completed | 654988 | 0 | False |
| Mini_n1000_ECS_p2 | completed | 654988 | 0 | False |
| Mini_n1000_TORCH_p0 | completed | 654988 | 0 | False |
| Mini_n1000_TORCH_p1 | completed | 654988 | 0 | False |
| Mini_n1000_TORCH_p2 | completed | 654988 | 0 | False |
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
| Nano_n1000_BASE_p0 | completed | 480268 | 0 | False |
| Nano_n1000_BASE_p1 | completed | 480268 | 0 | False |
| Nano_n1000_BASE_p2 | completed | 480268 | 0 | False |
| Nano_n1000_ECS_p0 | completed | 480268 | 0 | False |
| Nano_n1000_ECS_p1 | completed | 480268 | 0 | False |
| Nano_n1000_ECS_p2 | completed | 480268 | 0 | False |
| Nano_n1000_TORCH_p0 | completed | 480268 | 0 | False |
| Nano_n1000_TORCH_p1 | completed | 480268 | 0 | False |
| Nano_n1000_TORCH_p2 | completed | 480268 | 0 | False |
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
| Neo_n1000_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n1000_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n1000_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n1000_ECS_RUNTIME_p0 | completed | 1124910 | 2 | False |
| Neo_n1000_ECS_RUNTIME_p1 | completed | 1124910 | 2 | False |
| Neo_n1000_ECS_RUNTIME_p2 | completed | 1124910 | 2 | False |
| Neo_n1000_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n1000_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n1000_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n1000_TORCH_p0 | completed | 1124910 | 0 | False |
| Neo_n1000_TORCH_p1 | completed | 1124910 | 0 | False |
| Neo_n1000_TORCH_p2 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n4096_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n4096_ECS_RUNTIME_p0 | completed | 1124910 | 2 | False |
| Neo_n4096_ECS_RUNTIME_p1 | completed | 1124910 | 2 | False |
| Neo_n4096_ECS_RUNTIME_p2 | completed | 1124910 | 2 | False |
| Neo_n4096_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n4096_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n4096_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n4096_TORCH_p0 | oom | 1124910 | 0 | False |
| Neo_n4096_TORCH_p1 | oom | 1124910 | 0 | False |
| Neo_n4096_TORCH_p2 | oom | 1124910 | 0 | False |
| Neo_n512_BASE_p0 | completed | 1124910 | 0 | False |
| Neo_n512_BASE_p1 | completed | 1124910 | 0 | False |
| Neo_n512_BASE_p2 | completed | 1124910 | 0 | False |
| Neo_n512_ECS_RUNTIME_p0 | completed | 1124910 | 2 | False |
| Neo_n512_ECS_RUNTIME_p1 | completed | 1124910 | 2 | False |
| Neo_n512_ECS_RUNTIME_p2 | completed | 1124910 | 2 | False |
| Neo_n512_ECS_p0 | completed | 1124910 | 0 | True |
| Neo_n512_ECS_p1 | completed | 1124910 | 0 | True |
| Neo_n512_ECS_p2 | completed | 1124910 | 0 | True |
| Neo_n512_TORCH_p0 | completed | 1124910 | 0 | False |
| Neo_n512_TORCH_p1 | completed | 1124910 | 0 | False |
| Neo_n512_TORCH_p2 | completed | 1124910 | 0 | False |
| Pro_n512_TORCH_p0 | oom | 22678990 | 0 | False |
| Pro_n64_BASE_p0 | completed | 22678990 | 0 | False |
| Pro_n64_ECS_p0 | completed | 22678990 | 0 | False |
| Pro_n64_ECS_p1 | completed | 22678990 | 0 | False |
| Pro_n64_TORCH_p0 | completed | 22678990 | 0 | False |
