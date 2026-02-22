# Calculate Model Deviation

## Theory

Model deviation $\epsilon_y$ is the standard deviation of properties $\boldsymbol y$ inferred by an ensemble of models $\mathcal{M}_ 1, \dots, \mathcal{M}_{n_m}$ that are trained by the same dataset(s) with the model parameters initialized independently.
The DeePMD-kit supports $\boldsymbol y$ to be the atomic force $\boldsymbol F_i$ and the virial tensor $\boldsymbol \Xi$.
The model deviation is used to estimate the error of a model at a certain data frame, denoted by $\boldsymbol x$, containing the coordinates and chemical species of all atoms.
We present the model deviation of the atomic force and the virial tensor

```math
    \epsilon_{\boldsymbol{F},i} (\boldsymbol x)=
    \sqrt{\langle \lVert \boldsymbol F_i(\boldsymbol x; \boldsymbol \theta_k)-\langle \boldsymbol F_i(\boldsymbol x; \boldsymbol \theta_k) \rangle \rVert^2 \rangle},
```

```math
    \epsilon_{\boldsymbol{\Xi},{\alpha \beta}} (\boldsymbol x)=
    \frac{1}{N} \sqrt{\langle ( {\Xi}_{\alpha \beta}(\boldsymbol x; \boldsymbol \theta_k)-\langle {\Xi}_{\alpha \beta}(\boldsymbol x; \boldsymbol \theta_k) \rangle )^2 \rangle},
```

where $\boldsymbol \theta_k$ is the parameters of the model $\mathcal M_k$, and the ensemble average $\langle\cdot\rangle$ is estimated by

```math
    \langle \boldsymbol y(\boldsymbol x; \boldsymbol \theta_k) \rangle
    =
    \frac{1}{n_m} \sum_{k=1}^{n_m} \boldsymbol y(\boldsymbol x; \boldsymbol \theta_k).
```

Small $\epsilon_{\boldsymbol{F},i}$ means the model has learned the given data; otherwise, it is not covered, and the training data needs to be expanded.
If the magnitude of $\boldsymbol F_i$ or $\boldsymbol \Xi$ is quite large,
a relative model deviation $\epsilon_{\boldsymbol{F},i,\text{rel}}$ or $\epsilon_{\boldsymbol{\Xi},\alpha\beta,\text{rel}}$ can be used instead of the absolute model deviation:

```math
    \epsilon_{\boldsymbol{F},i,\text{rel}}  (\boldsymbol x)
    =
    \frac{\lvert \epsilon_{\boldsymbol{F},i} (\boldsymbol x) \lvert}
    {\lvert \langle \boldsymbol F_i (\boldsymbol x; \boldsymbol \theta_k) \rangle \lvert + \nu},
```

```math
    \epsilon_{\boldsymbol{\Xi},\alpha\beta,\text{rel}}  (\boldsymbol x)
    =
    \frac{ \epsilon_{\boldsymbol{\Xi},\alpha\beta} (\boldsymbol x) }
    {\lvert \langle \boldsymbol \Xi (\boldsymbol x; \boldsymbol \theta_k) \rangle \lvert + \nu},
```

where $\nu$ is a small constant used to protect
an atom where the magnitude of $\boldsymbol{F}_i$ or $\boldsymbol{\Xi}$ is small from having a large model deviation.

Statistics of $\epsilon_{\boldsymbol{F},i}$ and $\epsilon_{\boldsymbol{\Xi},{\alpha \beta}}$ can be provided, including the maximum, average, and minimal model deviation over the atom index $i$ and over the component index $\alpha,\beta$, respectively.
The maximum model deviation of forces $\epsilon_{\boldsymbol F,\text{max}}$ in a frame was found to be the best error indicator in a concurrent or active learning algorithm.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

One can also use a subcommand to calculate the deviation of predicted forces or virials for a bunch of models in the following way:

```bash
dp model-devi -m graph.000.pb graph.001.pb graph.002.pb graph.003.pb -s ./data -o model_devi.out
```

where `-m` specifies model files to be calculated, `-s` gives the data to be evaluated, `-o` the file to which model deviation results are dumped. Here is more information on this sub-command:

```{program-output} dp model-devi -h

```

For more details concerning the definition of model deviation and its application, please refer to [Yuzhi Zhang, Haidi Wang, Weijie Chen, Jinzhe Zeng, Linfeng Zhang, Han Wang, and Weinan E, DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models, Computer Physics Communications, 2020, 253, 107206.](https://doi.org/10.1016/j.cpc.2020.107206)

## Relative model deviation

By default, the model deviation is output in absolute value. If the argument `--relative` is passed, then the relative model deviation of the force will be output, including values output by the argument `--atomic`. The relative model deviation of the force on atom $i$ is defined by

$$E_{f_i}=\frac{\left|D_{f_i}\right|}{\left|f_i\right|+l}$$

where $D_{f_i}$ is the absolute model deviation of the force on atom $i$, $f_i$ is the norm of the force and $l$ is provided as the parameter of the keyword `relative`.
If the argument `--relative_v` is set, then the relative model deviation of the virial will be output instead of the absolute value, with the same definition of that of the force:

$$E_{v_i}=\frac{\left|D_{v_i}\right|}{\left|v_i\right|+l}$$
