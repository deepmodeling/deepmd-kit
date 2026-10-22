# 周期性 LES / SOG 长程能量：公式与 k 点实现方案

> 分支：`dpa4c+period`
> 目标：给 `deepmd/dpmodel/long_range`（latent_charge 模块）的 LES 和 SOG 两种
> kernel 增加周期边界支持，第一版只覆盖 **Python 训练路径（graph lower）**；
> 压缩 / CUDA direct-sum / C++ / LAMMPS 路径显式拒绝。
>
> 设计取向：**纯倒易空间（all-reciprocal）方案**。SOG 高斯核的傅里叶变换是
> 带本征高斯阻尼的整函数，实空间 erfc 残差恒为零，因此不需要 Ewald β 分离、
> 不需要实空间项、不需要动 edge 图。这与 `deepmd-kit-charge` 仓库
> （`sog_model._compute_periodic`）的路线一致，但修正其三处一致性问题：
> ① k 空间振幅缺 `(2π)^{3/2}σ³` 因子；② 周期分支丢失 LES；③ 自能用截断 k 和近似。

## 1. 记号

- 原子数 $N$，latent 通道数 $C$，电荷 $q_i^c$（$c=1..C$，单位 $e$）
- $k_e = 14.3996454784255$ eV·Å（`ElectrostaticConvertion`，`source/lib/include/ewald.h:19`）
- box 矩阵 $B$（行向量为 $\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3$），体积 $V = |\det B| > 0$
- 倒易格点：$\mathbf{k}(\mathbf{n}) = 2\pi B^{-T}\mathbf{n}$，$\mathbf{n}\in\mathbb{Z}^3$
- 每通道结构因子：$\hat\rho_c(\mathbf{k}) = \sum_i q_i^c\, e^{i\mathbf{k}\cdot\mathbf{r}_i}$
  （整体平移只给 $\hat\rho$ 乘一个模 1 相位，$|\hat\rho|^2$ 不变，坐标原点任意）
- SOG 参数：宽度几何序列 $\sigma_m = \sigma_0 b^m$（$m=0..M-1$），
  **修正后振幅** $a_m = \sqrt{2/\pi}\,\ln b/\sigma_m$
  （本仓库 `16f40a2b`；旧常数振幅 $4\pi\ln b$ 给出对数形核，已废弃）

## 2. 非周期公式（现有实现，不变）

$$E_{LR} = k_e \sum_{i<j}\sum_{c=1}^C q_i^c q_j^c\, K(r_{ij}),\qquad
K_{LES}(r) = \frac{\mathrm{erf}(\alpha r)}{r},\qquad
K_{SOG}(r) = \sum_{m=0}^{M-1} a_m\, e^{-r^2/2\sigma_m^2}$$

## 3. 周期公式（本方案）

### 3.1 核的傅里叶变换（配置单选：一个模型只用一种 kernel）

`long_range.kernel` 配置是互斥单选（`{"type": "les"}` 或 `{"type": "sog"}`，
`LatentChargeLongRange` 只持有一个 `BaseInteractionKernel`），
**不存在两种核相加的模型**。周期公式按 kernel 类型分别给出：

| | $\hat K(\mathbf{k})$ | $K(0)$（自接触值） | 有效 k 空间阻尼 |
|---|---|---|---|
| **LES** | $\dfrac{4\pi}{k^2}\, e^{-k^2/4\alpha^2}$ | $\dfrac{2\alpha}{\sqrt{\pi}}$ | $e^{-k^2/4\alpha^2}$（α 提供） |
| **SOG** | $\displaystyle\sum_{m=0}^{M-1} 4\pi\ln b\; \sigma_m^2\; e^{-\sigma_m^2 k^2/2}$ | $\displaystyle\sum_{m=0}^{M-1} a_m$ | $e^{-\sigma_0^2 k^2/2}$（最窄高斯提供） |

两行的来源：
- LES：$K(r)=\mathrm{erf}(\alpha r)/r$ 是宽度 $1/(\sqrt2\alpha)$ 的高斯模糊库仑势，
  FT 即阻尼库仑 $4\pi e^{-k^2/4\alpha^2}/k^2$；
- SOG：各项 FT 为 $a_m(2\pi)^{3/2}\sigma_m^3 e^{-\sigma_m^2 k^2/2}$
  $= 4\pi\ln b\,\sigma_m^2 e^{-\sigma_m^2 k^2/2}$（代入修正振幅
  $a_m=\sqrt{2/\pi}\ln b/\sigma_m$），求和即得表中式。

SOG 的一致性闭环（梯形离散，$\ln\sigma$ 等距、间距 $\ln b$）：

$$\sum_m 4\pi\ln b\,\sigma_m^2 e^{-\sigma_m^2 k^2/2}
\;\approx\; 4\pi\!\int_0^\infty\! \sigma^2 e^{-\sigma^2 k^2/2}\, d\ln\sigma
= \frac{4\pi}{k^2}$$

即修正振幅后的 SOG 在 k 空间**自动给出正确的未阻尼库仑倒易**，无需任何拟合。
这正是"不需要 Ewald β"的原因：高斯族核的每个模式自带 k 空间截断因子。

#### 3.1.1 系数对应关系（看似不同，实为 FT 对，无新增系数）

非周期实空间与周期 k 空间的 SOG 系数表面上一个是 $1/\sigma_m$、一个是
$\sigma_m^2$，但它们是**同一组参数** $(\sigma_0, b, M)$ 经 FT 严格联系的对：

$$\tilde a_m = a_m \cdot \underbrace{(2\pi)^{3/2}\sigma_m^3}_{\exp(-r^2/2\sigma^2)\ \text{的 FT 因子}}
= \frac{\sqrt{2/\pi}\,\ln b}{\sigma_m}\cdot(2\pi)^{3/2}\sigma_m^3
= 4\pi\ln b\,\sigma_m^2 \quad\left(\sqrt{2/\pi}\,(2\pi)^{3/2} = 4\pi\right)\checkmark$$

两条独立的一致性闭环，各自离散化出 $1/r$ 且互为 FT：

**实空间**：$\displaystyle \frac{1}{r} = \sqrt{\frac{2}{\pi}}\!\int_0^\infty\! e^{-r^2/2\sigma^2}\frac{d\sigma}{\sigma^2}
\xrightarrow{\ln\sigma\ \text{梯形}}\ \sum_m \frac{\sqrt{2/\pi}\ln b}{\sigma_m} e^{-r^2/2\sigma_m^2} = \sum_m a_m e^{-r^2/2\sigma_m^2}$ ✓

**k 空间**：$\displaystyle \frac{4\pi}{k^2} = 4\pi\!\int_0^\infty\! \sigma^2 e^{-\sigma^2k^2/2}\, d\ln\sigma
\xrightarrow{\ln\sigma\ \text{梯形}}\ 4\pi\ln b\sum_m \sigma_m^2 e^{-\sigma_m^2k^2/2} = \sum_m \tilde a_m e^{-\sigma_m^2k^2/2}$ ✓

注意与 `deepmd-kit-charge` 仓库的区别：那边的 `_compute_periodic` 直接把实空间
`amp` 填进 `kfac = amp·e^{-σ²k²/2}`，**漏了 $(2π)^{3/2}σ_m^3$ 因子**，导致周期
能量并非实空间核的点阵和，靠可训练 amp 硬拟合补偿；本方案显式补上，
修正振幅下恰好化简为 $4\pi\ln b\,\sigma_m^2$。

### 3.2 总能量

$$E_{LR}^{PBC} = k_e\left[\;
\frac{1}{2V}\sum_{c=1}^C\sum_{\mathbf{k}\neq\mathbf{0}} \hat K(\mathbf{k})\,\big|\hat\rho_c(\mathbf{k})\big|^2
\;-\;\frac{1}{2}\,K(0)\sum_{i,c}\big(q_i^c\big)^2
\;\right]$$

- 自接触值 $K(0)$ 按 §3.1 表取用（解析式，不要用截断 k 和近似）。
  注意 $K(0)=\sum_m a_m$ 用的仍是 §3.1.1 的同一组 $a_m$，不是新系数——
  表面差异只在自能的处理：非周期路径是 $i<j$ 求和（自身对天然不存在），
  周期 k 和经 Poisson 求和包含自身，故显式减掉 $\frac{1}{2}K(0)\sum q^2$。

- 减自项的来源：k 和通过 Poisson 求和包含全部 $i=j$（含自身镜像），
  其中 $\mathbf{n}=\mathbf{0}$ 的自身项是 $\frac{1}{2}K(0)\sum q^2$，
  必须显式减掉；自身镜像（$\mathbf{n}\neq 0$）是合法贡献，保留。
- **截断误差指数可控**：$k_{max}$ 处 LES 残差 ~ $e^{-k_{max}^2/4\alpha^2}$，
  SOG 残差 ~ $e^{-\sigma_0^2 k_{max}^2/2}$。

### 3.3 边界条件与净电荷（v1 决策）

| 项 | v1 处理 | 说明 |
|---|---|---|
| k = 0 模式 | 排除 | 条件收敛部分；$\mathbf{k}\neq\mathbf{0}$ 求和 = tin-foil（导体）边界 + 每通道均匀中和背景 |
| 净电荷 $Q_c \neq 0$ | **直接报错**，要求所有通道中性 | `charge_constraint` 默认 `neutral`，天然满足；背景自能修正 $-\pi k_e Q_c^2/(2\beta^2 V)$ 依赖单一阻尼宽度，多宽度下无唯一写法，列为开放项 |
| 真空表面项 $\frac{2\pi k_e}{3V}\lvert\sum_i q_i^c \mathbf{r}_i\rvert^2$ | 不加（tinfoil 默认） | 作为配置开关预留，与 `source/lib/src/ewald.cc` 行为一致 |

### 3.4 力

$$F_{i} = -\frac{\partial E}{\partial \mathbf{r}_i}
= -\frac{k_e}{V}\sum_{\mathbf{k}\neq 0} \hat K(\mathbf{k})\,\mathbf{k}
  \sum_{c=1}^C q_i^c\,\mathrm{Im}\!\left[\hat\rho_c(\mathbf{k})\, e^{-i\mathbf{k}\cdot\mathbf{r}_i}\right]$$

实现上走 autograd（`coord` 作 leaf，见 §5.3），上式用于对账测试。

### 3.5 位力（解析式，不用对 k 或 box autograd）

按 kernel 类型（把 $\hat K$ 写成模式 $W_j$ 之和：LES 单模式
$W_{LES}=\frac{4\pi}{k^2}e^{-k^2/4\alpha^2}$；SOG 有 $M$ 个
$W_{SOG,m}=4\pi\ln b\,\sigma_m^2 e^{-\sigma_m^2 k^2/2}$）：

$$\kappa(\mathbf{k}) = \frac{1}{\hat K(\mathbf{k})}\sum_j W_j(k)\,\big(2\delta_j + \gamma_j k^2\big),
\qquad \delta_{LES}=1,\ \gamma_{LES}=\tfrac{1}{2\alpha^2},\qquad
\delta_{SOG,m}=0,\ \gamma_{SOG,m}=\sigma_m^2$$

即 **LES**：$\kappa(\mathbf{k}) = 2 + \dfrac{k^2}{2\alpha^2}$；
**SOG**：$\kappa(\mathbf{k}) = \dfrac{\sum_m 4\pi\ln b\,\sigma_m^4 k^2\,
 e^{-\sigma_m^2 k^2/2}}{\hat K(\mathbf{k})}$（$\sigma_m^4$ 加权的模式平均）。
与 `source/lib/src/ewald.cc:224-235` 的单一 β 公式一致（LES 情形逐分量对上）：

$$\boxed{\;W_{rec} = \frac{k_e}{2V}\sum_c\sum_{\mathbf{k}\neq 0}
\hat K(\mathbf{k})\,\big|\hat\rho_c(\mathbf{k})\big|^2
\left[\mathbb{1} - \kappa(\mathbf{k})\,\hat{\mathbf{k}}\hat{\mathbf{k}}^{\top}\right]\;}$$

（应变求导时 mask 边界的变化为零测度贡献，忽略——标准做法。）

### 3.6 原子能量归属

$E_{rec}$ 是帧全局量。训练需要 per-atom energy：每帧 $E_{rec}/N_f$ 均摊
（与现有非周期路径"pair 能量半归两端"不同，但 force/virial 不受影响；
out-bias 统计看到的是完整能量，语义正确）。

## 4. 与非周期路径的关系

- `box is None` → 走现有 `all_pairs` 实空间路径（不动）；
- `box is not None` → 走本方案（k 空间），**且不再要求 `all_pairs` 拓扑**；
- 周期性状的判定沿用 `check_complete_frames` 的反向逻辑：非 None box 从
  报错改为分支。模型级声明建议：`required_topologies()` 返回
  `("k_grid",)`（周期）或 `("all_pairs",)`（非周期），按配置在构造时确定，
  一个模型实例两种模式都支持（逐帧 box 判定）。
- $V\to\infty$ 极限：k 和变积分，自动退化——无周期/非周期物理跳变
  （修复 `deepmd-kit-charge` 仓库周期分支丢 LES 导致的能力降级）。

## 5. k 点的程序实现

### 5.1 网格生成（eager，编译区外）

输入：每帧 box $B$、kernel 参数 $\alpha,\sigma_0$、容差 $\varepsilon$（默认 $10^{-8}$）。

```python
def build_k_grid(box, alpha=None, sigma0=None, eps=1e-8):
    # (1) 截断半径：按 kernel 类型取对应阻尼的
    if kernel_type == "les":                        # exp(-k^2/4a^2) < eps
        k_max = 2.0 * alpha * sqrt(log(1/eps))
    else:                                           # "sog": exp(-s0^2 k^2/2) < eps
        k_max = sqrt(2.0 * log(1/eps)) / sigma0
    # (2) 倒易基矢（列）与每维保守整数界
    #     |k|<=k_max => |n_d| <= k_max*|a_d|/(2pi)  (Cauchy-Schwarz)
    # 倒易基矢（列）：g_d = 2*pi*B^{-1}[:, d]，使 a_d · g_e = 2pi delta_de
    # 注意是 inv(B) 而非 inv(B).T（B 非对称时 transpose 是错的）
    rec = 2*pi * inv(box)                            # rec[:, d] = g_d
    n_d = ceil(k_max * norm(box, axis=1) / (2*pi))   # (3,)
    # (3) 枚举 + 字典序（确定性顺序！）+ 球形截断
    ks, ns = [], []
    for n0 in range(-n_d[0], n_d[0]+1):              # 外层→内层固定序
      for n1 in range(-n_d[1], n_d[1]+1):
        for n2 in range(-n_d[2], n_d[2]+1):
          if n0 == n1 == n2 == 0: continue
          k = rec @ (n0, n1, n2)
          if dot(k, k) <= k_max*k_max: keep
    return k_vec (nk,3), k2 (nk,)
```

- **数量级**：$\sigma_0 = 2.18$ Å、$\varepsilon=10^{-8}$ → $k_{max}\approx 6.2$ Å⁻¹；
  15 Å 立方盒约 $9^3-1 \approx 700$ 个 k 点；$\hat\rho$ 计算量 $O(N_k\cdot N\cdot C)$。
- 每帧网格不同（box 不同）→ batch 拼接时需要逐帧归属（§5.2）。
- **compile：ragged 动态轴**（`nk` 成为第四个符号轴，照 `npair` 协议，
  flatten/unflatten 为 k_grid 携 `(frame, vec, mask, volume, coord, strain)`）。
- **固定网格 + mask（已实现，`k_grid_mode: fixed`）**：v1 时未采用，后按需求
  补上作为可选模式。整数网格 $\mathbf{n} \in [-n_d, n_d]^3$（不含零矢量）
  由数据统计中最大 cell 的逐方向界 $n_d = \lceil k_{max}|a_d|/2\pi \rceil$
  一次性枚举（`enumerate_k_grid_bounds`，可用 `k_grid_bounds` 配置覆盖），
  存入模型并随其序列化；每个 batch 以静态形状 matmul 求各帧波矢
  $\mathbf{k} = 2\pi B_f^{-1}\mathbf{n}$（k 轴长度固定为 $n_f \times n_k$，无
  guard 点），帧达不到的点（$|k| > k_{max}$，$k_{max}$ 取构建时 kernel 参数）
  由 mask 屏蔽（`build_fixed_k_grid`）。网格是硬性的：cell 超出网格容量或
  kernel 漂移使 $k_{max}$ 越界时，该帧仅以网格内 k 点截断求和（倒格点欠采样的
  近似，力和位力仍是该近似能量的精确导数），并发 warning。零矢量不入网格：LES 的 $\hat K$ 在
  $k=0$ 奇异，masked 的 inf 会在反向传播产生 NaN。开销约为 ragged 球内点数
  的 ~2 倍（各向同性 cell），换来完全静态的形状。
- **解耦开关（`decouple_k_grid: true`）**：默认（耦合）每 batch 以当前 kernel
  参数现算 $k_{max}$，kernel 漂移 → 网格跟随，能量定义始终是当前 kernel 的收敛
  能量；置 true 后 $k_{max}$ 在构造时按初始参数冻结（`frozen_k_max`，随模型
  序列化），kernel 漂移不再影响 k 点选择，图内 $\hat K(k)$ 仍用活参数求值
  （梯度不受影响）。两种模式共用；fixed 模式下解耦后 mask 也不再随漂移收紧。

### 5.2 图内计算（伪代码，全部纯张量运算，无 data-dependent 控制流）

```python
# 输入：coord (N,3) [leaf], strain (nf,3,3) [全零 leaf], q (N,C) [电荷网络输出],
#       k_vec (nk,3), k_frame (nk,), k_mask (nk,), volume (nf,),
#       node_frame (N,)
# 应变 leaf 技巧：位力 = -dE/d(strain)，不再需解析位力公式
S        = I + strain
coord'   = S[node_frame] @ coord            # 每节点 (3,3)@(3,)
k'       = inv(S)[k_frame]^{-T} @ k_vec     # 每 k 点 (3,3)@(3,)
volume'  = volume * det(S)
kr       = k' @ coord'.T                    # (nk, N)
same     = (node_frame[None, :] == k_frame[:, None])     # (nk, N) 帧归属
rho_r    = (cos(kr) * same) @ q             # (nk, C) segment sum
rho_i    = (sin(kr) * same) @ q             # (nk, C)
s_sq     = (rho_r**2 + rho_i**2).sum(-1)    # (nk,)

# 模式权重（alpha 或 sigma 可训练，梯度照常；按 kernel 类型二选一）
if kernel_type == "les":
    K_hat = 4*pi * exp(-k2' / (4*alpha**2)) / k2'         # (nk,)
    K0    = 2*alpha / sqrt(pi)
else:  # "sog"
    K_hat = sum_m( amp[m]*(2pi)**1.5*sigma[m]**3          # (nk,)，模式循环展开：
                   * exp(-0.5*sigma[m]**2*k2') )          #  避免 (nk,M) 中间张量
    K0    = amp.sum()                                     #  （inductor 在其二次反传
                                                          #   里生成 NaN；LES 不受影响）

# 能量（自能项只作用于有真实 k 点的帧——周期帧）
E_rec  = k_e * segment_sum(k_mask * K_hat * s_sq / (2*volume'[k_frame]),
                           k_frame, nf)
E_self = -0.5 * k_e * K0 * (q**2).sum(-1) * is_periodic_frame[node_frame]
```

- 精度策略（对齐仓库惯例）：`rho_r/rho_i` 与 k 求和用 float64 累加，
  k 点按 §5.1 固定字典序 → 完全确定性。
- 力：对 `coord` leaf autograd（框架内 `fit_output_to_model_output_graph`
  的 leaf 列表扩展，与 `edge_vec`/`pair_vec` 并列）。
- 位力：§3.5 解析式，tensor 实现，`kappa(k)` 对可训练 $\alpha,\sigma$ 可导。

### 5.3 图 lower 签名影响（本方案真正动架构的一处）

现有 graph lower **没有坐标输入**。$e^{i\mathbf{k}\cdot\mathbf{r}_i}$ 需要
绝对坐标相位。最终实现采用：**`coord` 和零应变 `strain` 都作为 k_grid 拓扑
的 flatten 输入**（`(frame, vec, mask, volume, coord, strain)`），在
`forward_common_lower_graph` 里 `.detach().requires_grad_(True)` 重 leaf——
反向传播必须根植在图输入上（make_fx 内部创建的 leaf 在 inductor 下产出
NaN）。力由 coord leaf 的梯度直接给出；位力 $W = -\partial E/\partial\varepsilon$
由 strain leaf 的梯度给出（模块内以 $r'=r(I+\varepsilon)^T$、
$k'=k(I+\varepsilon)^{-1}$、$V'=V\det(I+\varepsilon)$ 求值）。
原子位力约定：$r\otimes F$ 部分逐节点，帧余量均摊。

## 6. 验证计划

1. **M1 对账脚本**（`verify_periodic_lr.py`，独立 torch float64，不接框架）——
   已实现并全部通过：
   - **LES**：k 空间公式 vs 独立双参数 Ewald 分解（实空间 `[erf(αr)−erf(βr)]/r`
     镜像和 + k 空间 β 项 + 自能），diff 1e-9；
   - **SOG 分模式 Poisson 恒等式**：每个高斯模式（σ ≤ 10 Å）实空间镜像和 vs
     k 空间，diff ~1e-10～1e-11（宽模式以同样的数学自然延伸）；
   - **SOG 复合对账**：全 k 空间 vs 窄模式（σ≤4 Å）实空间镜像和 + 宽模式 k 空间，
     diff 1e-8——注意**不要**用 `[K_sog−erf(βr)/r]` 做参考：该差核在
     $r \gtrsim \sigma_{max}\approx 610$ Å 处残余 $-1/r$ 尾巴，壳层求和无法收敛；
   - 力（autograd vs FD）~1e-8，位力（解析式 §3.5 vs 应变 FD）~1e-7；
   - k_max 收敛表：两种 kernel 均指数收敛；
   - 纯 LES 可用 `deepmd/tf/infer/ewald_recp.py` 交叉验证（其 β = 本方案 α，
     只含倒易项）——需要 C++ TF 算子库构建，未在 M1 中执行。
2. **M2 框架接入**（已完成）：`LatentChargeLongRange` 周期分支 +
   k 网格 eager 构建 + `coord`/`strain` 双 leaf；`direct_sum_eligible()`
   不变（压缩路径在 `enable_compression` 拒绝周期模块）；导出/冻结拒绝。
   单测：`source/tests/pt_expt/model/test_long_range.py`（周期组，14 个）
   与 `source/tests/common/dpmodel/test_long_range.py`（k 网格/kernel 原语）。
   实测（12 原子三斜盒）：能量分解 vs M1 参考 2e-7（LES）/1e-6（SOG），
   力 FD 3.6e-7/1.5e-8，位力 FD 3e-7/9.8e-9，编译 == eager 至 1e-13，
   Hessian（图路径）vs FD 2e-6。
3. 拒绝面：周期 + 压缩/CUDA/LAMMPS/非中性通道 → 显式报错。

## 8. 实现踩坑记录

- **倒易格点约定**：box 行向量是晶格向量 $\mathbf{a}_d$ 时，倒易格点
  $\mathbf{k} = 2\pi\,\mathbf{B}^{-1}\mathbf{n}$（**没有转置**）。写错成
  $2\pi B^{-T}\mathbf{n}$ 在正交/立方盒上碰巧正确，三斜盒上 k 点偏离格点
  达 0.5·$2\pi/|a|$，造成 0.5 eV 量级的能量错误。验证方法：
  `B @ rec / (2*pi) == I`（必须逐元素成立）。
- **SOG 的 (nk, M) 中间张量**：inductor 对该广播的二次反传生成 NaN
  （make_fx 单独正确、LES 不受影响）；按模式 Python 循环展开为 (nk,) 项
  求和后正常。
- **内部创建的 autograd leaf**：在 make_fx 图内 `new_zeros().requires_grad_()`
  的叶子（strain）在 inductor 下梯度为 NaN；叶子必须作为 flatten 输入传入
  再在图内 re-leaf（`.detach().requires_grad_()`），与 edge_vec 同模式。
- **自能项的帧归属**：$-\frac{1}{2}K(0)\sum q^2$ 只作用于有真实 k 点的帧
  （周期帧）；非周期批在周期模型上跑时 k_grid 全 guard，自能必须跳过，
  否则能量错 $O(K(0)\sum q^2)$。
- **FD 位力应变的约定**：坐标与 box 必须同族变形（$r\to(I+\varepsilon)r$，
  $B\to B(I+\varepsilon)^T$），混用（`d @ box`）破坏旋转不变性。
- **近截断原子对的 FD Hessian**：原子跨越 rcut 时力有拐点，FD Hessian 在
  任意步长下都停在 O(0.1) 误差——测试体系必须让所有原子远离 cutoff 边界
  （紧致簇 + 大盒）。

## 7. 开放项（不阻塞 v1）

- 非中性通道的背景自能修正（多阻尼宽度下需选定 regularization）；
- 真空表面项配置开关；
- box 在大范围内变化的训练集的 ragged k 轴（v2）；
- 压缩路径 / CUDA 周期 direct-sum / LAMMPS 周期部署；
- Hessian 与周期 k 项的兼容性检查。
