# Compress a model {{ tensorflow_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

## Theory

The compression of the DP model uses three techniques, tabulated inference, operator merging, and precise neighbor indexing, to improve the performance of model training and inference when the model parameters are properly trained.

For better performance, the NN inference can be replaced by tabulated function evaluations if the input of the NN is of dimension one.
The idea is to approximate the output of the NN by a piece-wise polynomial fitting.
The input domain (a compact domain in $\mathbb R$) is divided into $L_c$ equally spaced intervals, in which we apply a fifth-order polynomial $g^l_m(x)$ approximation of the $m$-th output component of the NN function:

```math
    g^l_m(x) = a^l_m x^5 + b^l_m x^4 + c^l_m x^3 + d^l_m x^2 + e^l_m x + f^l_m,\quad
    x \in [x_l, x_{l+1}),
```

where $l=1,2,\dots,L_c$ is the index of the intervals, $x_1, \dots, x_{L_c}, x_{L_c+1}$ are the endpoints of the intervals, and $a^l_m$, $b^l_m$, $c^l_m$, $d^l_m$, $e^l_m$, and $f^l_m$ are the fitting parameters.
The fitting parameters can be computed by the equations below:

```math
    a^l_m = \frac{1}{2\Delta x_l^5}[12h_{m,l}-6(y'_{m,l+1}+y'_{m,l})\Delta x_l + (y''_{m,l+1}-y''_{m,l})\Delta x_l^2],
```

```math
    b^l_m = \frac{1}{2\Delta x_l^4}[-30h_{m,l} +(14y'_{m,l+1}+16y'_{m,l})\Delta x_l + (-2y''_{m,l+1}+3y''_{m,l})\Delta x_l^2],
```

```math
    c^l_m = \frac{1}{2\Delta x_l^3}[20h_{m,l}-(8y'_{m,l+1}+12y'_{m,l})\Delta x_l + (y''_{m,l+1}-3y''_{m,l})\Delta x_l^2],
```

```math
    d^l_m = \frac{1}{2}y''_{m,l},
```

```math
    e^l_m = y_{m,l}',
```

```math
    f^l_m = y_{m,l},
```

where $\Delta x_l=x_{l+1}-x_l$ denotes the size of the interval. $h_{m,l}=y_{m,l+1}-y_{m,l}$. $y_{m,l} = y_m(x_l)$, $y'_{m,l} = y'_m(x_l)$ and $y''_{m,l} = y''_m(x_l)$ are the value, the first-order derivative, and the second-order derivative of the $m$-th component of the target NN function at the interval point $x_l$, respectively.
The first and second-order derivatives are easily calculated by the back-propagation of the NN functions.

In the standard DP model inference, taking the [two-body embedding descriptor](../model/train-se-e2-a.md) as an example, the matrix product $(\mathcal G^i)^T \mathcal R$ requires the transfer of the tensor $\mathcal G^i$ between the register and the host/device memories, which usually becomes the bottle-neck of the computation due to the relatively small memory bandwidth of the GPUs.
The compressed DP model merges the matrix multiplication $(\mathcal G^i)^T \mathcal R$ with the tabulated inference step.
More specifically, once one column of the $(\mathcal G^i)^T$ is evaluated, it is immediately multiplied with one row of the environment matrix in the register, and the outer product is deposited to the result of $(\mathcal G^i)^T \mathcal R$.
By the operator merging technique, the allocation of $\mathcal G^i$ and the memory movement between register and host/device memories is avoided.
The operator merging of the three-body embedding can be derived analogously.

The first dimension, $N_c$, of the environment ($\mathcal R^i$) and embedding ($\mathcal G^i$) matrices is the expected maximum number of neighbors.
If the number of neighbors of an atom is smaller than $N_c$, the corresponding positions of the matrices are pad with zeros.
In practice, if the real number of neighbors is significantly smaller than $N_c$, a notable operation is spent on the multiplication of padding zeros.
In the compressed DP model, the number of neighbors is precisely indexed at the tabulated inference stage, further saving computational costs.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

Once the frozen model is obtained from DeePMD-kit, we can get the neural network structure and its parameters (weights, biases, etc.) from the trained model, and compress it in the following way:

```bash
dp compress -i graph.pb -o graph-compress.pb
```

where `-i` gives the original frozen model, `-o` gives the compressed model. Several other command line options can be passed to `dp compress`, which can be checked with

```bash
$ dp compress --help
```

An explanation will be provided

```
usage: dp compress [-h] [-v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-l LOG_PATH]
                   [-m {master,collect,workers}] [-i INPUT] [-o OUTPUT]
                   [-s STEP] [-e EXTRAPOLATE] [-f FREQUENCY]
                   [-c CHECKPOINT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  -v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR,
                        1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -l LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not
                        specified, the logs will only be output to console
                        (default: None)
  -m {master,collect,workers}, --mpi-log {master,collect,workers}
                        Set the manner of logging when running with MPI.
                        'master' logs only on main process, 'collect'
                        broadcasts logs from workers to master and 'workers'
                        means each process will output its own log (default:
                        master)
  -i INPUT, --input INPUT
                        The original frozen model, which will be compressed by
                        the code (default: frozen_model.pb)
  -o OUTPUT, --output OUTPUT
                        The compressed model (default:
                        frozen_model_compressed.pb)
  -s STEP, --step STEP  Model compression uses fifth-order polynomials to
                        interpolate the embedding-net. It introduces two
                        tables with different step size to store the
                        parameters of the polynomials. The first table covers
                        the range of the training data, while the second table
                        is an extrapolation of the training data. The domain
                        of each table is uniformly divided by a given step
                        size. And the step(parameter) denotes the step size of
                        the first table and the second table will use 10 *
                        step as it's step size to save the memory. Usually the
                        value ranges from 0.1 to 0.001. Smaller step means
                        higher accuracy and bigger model size (default: 0.01)
  -e EXTRAPOLATE, --extrapolate EXTRAPOLATE
                        The domain range of the first table is automatically
                        detected by the code: [d_low, d_up]. While the second
                        table ranges from the first table's upper
                        boundary(d_up) to the extrapolate(parameter) * d_up:
                        [d_up, extrapolate * d_up] (default: 5)
  -f FREQUENCY, --frequency FREQUENCY
                        The frequency of tabulation overflow check(Whether the
                        input environment matrix overflow the first or second
                        table range). By default do not check the overflow
                        (default: -1)
  -c CHECKPOINT_FOLDER, --checkpoint-folder CHECKPOINT_FOLDER
                        path to checkpoint folder (default: .)
  -t TRAINING_SCRIPT, --training-script TRAINING_SCRIPT
                        The training script of the input frozen model
                        (default: None)
```

**Parameter explanation**

Model compression, which includes tabulating the embedding net.
The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. For model descriptor with `se_e2_a` type, the first sub-table takes the stride(parameter) as its uniform stride, while the second sub-table takes 10 _ stride as its uniform stride; For model descriptor with `se_e3` type, the first sub-table takes 10 _ stride as it's uniform stride, while the second sub-table takes 100 _ stride as it's uniform stride.
The range of the first table is automatically detected by DeePMD-kit, while the second table ranges from the first table's upper boundary(upper) to the extrapolate(parameter) _ upper.
Finally, we added a check frequency parameter. It indicates how often the program checks for overflow(if the input environment matrix overflows the first or second table range) during the MD inference.

**Justification of model compression**

Model compression, with little loss of accuracy, can greatly speed up MD inference time. According to different simulation systems and training parameters, the speedup can reach more than 10 times at both CPU and GPU devices. At the same time, model compression can greatly change memory usage, reducing as much as 20 times under the same hardware conditions.

**Acceptable original model version**

The model compression interface requires the version of DeePMD-kit used in the original model generation should be `2.0.0-alpha.0` or above. If one has a frozen 1.2 or 1.3 model, one can upgrade it through the `dp convert-from` interface. (eg: `dp convert-from 1.2/1.3 -i old_frozen_model.pb -o new_frozen_model.pb`)

**Acceptable descriptor type**

Descriptors with `se_e2_a`, `se_e3`, `se_e2_r` and `se_atten_v2` types are supported by the model compression feature. `Hybrid` mixed with the above descriptors is also supported.

Notice: Model compression for the `se_atten_v2` descriptor is exclusively designed for models with the training parameter {ref}`attn_layer <model/descriptor[se_atten_v2]/attn_layer>` set to 0.

**Available activation functions for descriptor:**

- tanh
- gelu
- relu
- relu6
- softplus
- sigmoid
