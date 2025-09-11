# DeepMD-kit 模型压缩功能详细分析

## 概述

DeepMD-kit 的 compress 功能是一种模型优化技术，通过**表格化推理**（tabulated inference）、**算子融合**（operator merging）和**精确邻域索引**（precise neighbor indexing）三种技术来提高模型的推理性能。这些技术共同作用，可以显著减少内存使用和计算开销，同时保持模型精度的损失在可接受范围内。

## 理论基础

### 表格化推理

压缩的核心思想是将神经网络推理替换为查表操作。对于输入维度为 1 的神经网络函数，可以使用分段多项式拟合来近似网络输出。

#### 五次多项式拟合

对于每个区间 $[x_l, x_{l+1})$，使用五次多项式来近似神经网络输出：

```math
g^l_m(x) = a^l_m x^5 + b^l_m x^4 + c^l_m x^3 + d^l_m x^2 + e^l_m x + f^l_m
```

多项式系数通过以下公式计算：

```math
a^l_m = \frac{1}{2\Delta x_l^5}[12h_{m,l}-6(y'_{m,l+1}+y'_{m,l})\Delta x_l + (y''_{m,l+1}-y''_{m,l})\Delta x_l^2]
```

```math
b^l_m = \frac{1}{2\Delta x_l^4}[-30h_{m,l} +(14y'_{m,l+1}+16y'_{m,l})\Delta x_l + (-2y''_{m,l+1}+3y''_{m,l})\Delta x_l^2]
```

```math
c^l_m = \frac{1}{2\Delta x_l^3}[20h_{m,l}-(8y'_{m,l+1}+12y'_{m,l})\Delta x_l + (y''_{m,l+1}-3y''_{m,l})\Delta x_l^2]
```

```math
d^l_m = \frac{1}{2}y''_{m,l}
```

```math
e^l_m = y'_{m,l}
```

```math
f^l_m = y_{m,l}
```

其中：

- $\Delta x_l = x_{l+1} - x_l$ 为区间长度
- $h_{m,l} = y_{m,l+1} - y_{m,l}$
- $y_{m,l} = y_m(x_l)$, $y'_{m,l} = y'_m(x_l)$, $y''_{m,l} = y''_m(x_l)$ 分别是函数值、一阶导数和二阶导数

## PyTorch 后端实现流程

### 总体架构

PyTorch 后端的 compress 功能主要涉及以下组件：

1. **入口函数**: `deepmd/pt/entrypoints/compress.py`
2. **核心算法**: `deepmd/pt/utils/tabulate.py` 中的 `DPTabulate` 类
3. **模型接口**: `deepmd/pt/model/model/make_model.py`
4. **描述符接口**: `deepmd/pt/model/descriptor/se_a.py`
5. **C++自定义 OP**: `source/op/pt/tabulate_multi_device.cc`

### 详细实现流程

#### 1. 入口函数执行

```python
def enable_compression(
    input_file: str,
    output: str,
    stride: float = 0.01,
    extrapolate: int = 5,
    check_frequency: int = -1,
    training_script: Optional[str] = None,
) -> None:
    # 1. 加载原始模型
    saved_model = torch.jit.load(input_file, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict())

    # 2. 计算最小邻域距离（如果需要）
    if model.get_min_nbor_dist() is None:
        # 从训练数据计算最小邻域距离
        # ... 计算逻辑

    # 3. 启用压缩
    model.enable_compression(
        extrapolate,
        stride,
        stride * 10,
        check_frequency,
    )

    # 4. 保存压缩模型
    model = torch.jit.script(model)
    torch.jit.save(model, output)
```

#### 2. 模型压缩启用

模型的 `enable_compression` 方法调用：

```python
def enable_compression(
    self,
    table_extrapolate: float = 5,
    table_stride_1: float = 0.01,
    table_stride_2: float = 0.1,
    check_frequency: int = -1,
) -> None:
    self.atomic_model.enable_compression(
        self.get_min_nbor_dist(),
        table_extrapolate,
        table_stride_1,
        table_stride_2,
        check_frequency,
    )
```

#### 3. 原子模型压缩启用

原子模型调用描述符的压缩方法：

```python
def enable_compression(
    self,
    min_nbor_dist: float,
    table_extrapolate: float = 5,
    table_stride_1: float = 0.01,
    table_stride_2: float = 0.1,
    check_frequency: int = -1,
) -> None:
    self.descriptor.enable_compression(
        min_nbor_dist,
        table_extrapolate,
        table_stride_1,
        table_stride_2,
        check_frequency,
    )
```

#### 4. 描述符压缩实现

SE_A 描述符的压缩实现：

```python
def enable_compression(
    self,
    min_nbor_dist: float,
    table_extrapolate: float = 5,
    table_stride_1: float = 0.01,
    table_stride_2: float = 0.1,
    check_frequency: int = -1,
) -> None:
    if self.compress:
        raise ValueError("Compression is already enabled.")

    data = self.serialize()
    self.table = DPTabulate(
        self,
        data["neuron"],
        data["type_one_side"],
        data["exclude_types"],
        ActivationFn(data["activation_function"]),
    )

    self.table_config = [
        table_extrapolate,
        table_stride_1,
        table_stride_2,
        check_frequency,
    ]

    self.lower, self.upper = self.table.build(
        min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
    )

    self.sea.enable_compression(
        self.table.data, self.table_config, self.lower, self.upper
    )
```

### DPTabulate 类详解

`DPTabulate` 类是压缩功能的核心，继承自 `BaseTabulate`。

#### 初始化过程

```python
def __init__(
    self,
    descrpt: Any,
    neuron: list[int],
    type_one_side: bool = False,
    exclude_types: list[list[int]] = [],
    activation_fn: ActivationFn = ActivationFn("tanh"),
) -> None:
    super().__init__(
        descrpt,
        neuron,
        type_one_side,
        exclude_types,
        True,  # is_pt=True
    )
    self.descrpt_type = self._get_descrpt_type()
    # ... 初始化各种参数
```

#### 构建表格过程

`build` 方法是表格化实现的核心：

```python
def build(
    self, min_nbor_dist: float, extrapolate: float, stride0: float, stride1: float
) -> tuple[dict[str, int], dict[str, int]]:
    # 1. 计算环境矩阵范围
    lower, upper = self._get_env_mat_range(min_nbor_dist)

    # 2. 根据描述符类型构建表格
    if self.descrpt_type == "A":
        # SE_A描述符的处理逻辑
        for ii in range(self.table_size):
            # 计算表格范围
            xx = np.arange(ll, uu, stride0, dtype=self.data_type)
            xx = np.append(xx, np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type))
            xx = np.append(xx, np.array([extrapolate * uu], dtype=self.data_type))

            # 构建子表格
            self._build_lower(net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline)

    # 3. 转换数据格式
    self._convert_numpy_to_tensor()
    return self.lower, self.upper
```

#### \_build_lower 方法详解

这个方法实现具体的表格构建算法：

```python
def _build_lower(
    self,
    net: int,
    xx: np.ndarray,
    idx: int,
    upper: float,
    lower: float,
    stride0: int,
    stride1: int,
    extrapolate: bool,
    nspline: int,
) -> None:
    # 1. 计算函数值、导数
    vv, dd, d2 = self._make_data(xx, idx)

    # 2. 初始化表格数据存储
    self.data[net] = np.zeros([nspline, 6 * self.last_layer_size], dtype=self.data_type)

    # 3. 计算区间长度
    tt = np.full((nspline, self.last_layer_size), stride1)
    tt[: int((upper - lower) / stride0), :] = stride0

    # 4. 计算函数值差分
    hh = vv[1 : nspline + 1, : self.last_layer_size] - vv[:nspline, : self.last_layer_size]

    # 5. 计算五次多项式系数
    # 常数项 f^l_m = y_{m,l}
    self.data[net][:, : 6 * self.last_layer_size : 6] = vv[:nspline, : self.last_layer_size]

    # 一次项系数 e^l_m = y'_{m,l}
    self.data[net][:, 1 : 6 * self.last_layer_size : 6] = dd[:nspline, : self.last_layer_size]

    # 二次项系数 d^l_m = 0.5 * y''_{m,l}
    self.data[net][:, 2 : 6 * self.last_layer_size : 6] = 0.5 * d2[:nspline, : self.last_layer_size]

    # 三次项系数 c^l_m
    self.data[net][:, 3 : 6 * self.last_layer_size : 6] = (
        1 / (2 * tt * tt * tt)
    ) * (
        20 * hh
        - (8 * dd[1 : nspline + 1, : self.last_layer_size] + 12 * dd[:nspline, : self.last_layer_size]) * tt
        - (3 * d2[:nspline, : self.last_layer_size] - d2[1 : nspline + 1, : self.last_layer_size]) * tt * tt
    )

    # 四次项系数 b^l_m
    self.data[net][:, 4 : 6 * self.last_layer_size : 6] = (
        1 / (2 * tt * tt * tt * tt)
    ) * (
        -30 * hh
        + (14 * dd[1 : nspline + 1, : self.last_layer_size] + 16 * dd[:nspline, : self.last_layer_size]) * tt
        + (3 * d2[:nspline, : self.last_layer_size] - 2 * d2[1 : nspline + 1, : self.last_layer_size]) * tt * tt
    )

    # 五次项系数 a^l_m
    self.data[net][:, 5 : 6 * self.last_layer_size : 6] = (
        1 / (2 * tt * tt * tt * tt * tt)
    ) * (
        12 * hh
        - 6 * (dd[1 : nspline + 1, : self.last_layer_size] + dd[:nspline, : self.last_layer_size]) * tt
        + (d2[1 : nspline + 1, : self.last_layer_size] - d2[:nspline, : self.last_layer_size]) * tt * tt
    )

    # 6. 记录边界信息
    self.upper[net] = upper
    self.lower[net] = lower
```

#### \_make_data 方法详解

这个方法通过前向传播计算函数值和导数：

```python
def _make_data(self, xx: np.ndarray, idx: int) -> Any:
    xx = torch.from_numpy(xx).view(-1, 1).to(env.DEVICE)

    for layer in range(self.layer_size):
        if layer == 0:
            # 第一层特殊处理
            xbar = torch.matmul(xx, torch.from_numpy(self.matrix["layer_" + str(layer + 1)][idx]).to(env.DEVICE))
            xbar += torch.from_numpy(self.bias["layer_" + str(layer + 1)][idx]).to(env.DEVICE)

            # 根据神经元数量选择处理方式
            if self.neuron[0] == 1:
                yy = self._layer_0(xx, self.matrix["layer_" + str(layer + 1)][idx], self.bias["layer_" + str(layer + 1)][idx])
                yy += xx
                dy = unaggregated_dy_dx_s(yy - xx, self.matrix["layer_" + str(layer + 1)][idx], xbar, self.functype)
                dy += torch.ones((1, 1), dtype=yy.dtype)
                dy2 = unaggregated_dy2_dx_s(yy - xx, dy, self.matrix["layer_" + str(layer + 1)][idx], xbar, self.functype)
            # ... 其他情况处理
        else:
            # 后续层处理
            # ... 类似的计算逻辑

    # 返回函数值、导数和二阶导数
    vv = zz.detach().cpu().numpy().astype(self.data_type)
    dd = dy.detach().cpu().numpy().astype(self.data_type)
    d2 = dy2.detach().cpu().numpy().astype(self.data_type)
    return vv, dd, d2
```

### 算子融合技术

算子融合是压缩功能的重要优化技术，它通过以下方式提高性能：

1. **矩阵乘法融合**: 将嵌入层的输出与环境矩阵的乘法合并到表格化过程中
2. **内存访问优化**: 避免了嵌入矩阵在寄存器和内存之间的频繁传输
3. **计算优化**: 减少了不必要的中间结果存储

#### 融合前后的对比

**传统方式**:

```python
# 1. 计算嵌入层输出 G
G = embedding_net(env_matrix)
# 2. 矩阵乘法 G^T @ R
result = G.T @ env_matrix
# 3. 存储中间结果G
```

**融合方式**:

```python
# 直接在表格化过程中完成乘法
result = tabulate_fusion(G, R)  # G的计算和乘法在同一个内核中完成
```

### 精确邻域索引技术

精确邻域索引通过以下方式优化性能：

1. **动态邻域数量**: 根据实际邻域数量而不是最大邻域数量进行计算
2. **减少无效计算**: 避免对填充的零值进行计算
3. **内存效率**: 减少内存使用和带宽消耗

### C++自定义算子实现

PyTorch 后端使用 C++自定义算子来实现高性能的表格化推理：

#### 主要函数接口

```cpp
template <typename FPTYPE>
void TabulateFusionSeAForward(
    const torch::Tensor& table_tensor,      // 表格数据
    const torch::Tensor& table_info_tensor, // 表格信息
    const torch::Tensor& em_x_tensor,       // 环境矩阵x分量
    const torch::Tensor& em_tensor,         // 环境矩阵
    const torch::Tensor& two_embed_tensor,  // 二体嵌入
    int64_t last_layer_size,               // 最后一层大小
    torch::Tensor& descriptor_tensor       // 输出描述符
);
```

#### CPU/GPU 统一接口

代码同时支持 CPU 和 GPU 计算：

```cpp
if (device == "GPU") {
    deepmd::tabulate_fusion_se_a_gpu(descriptor, table, table_info, em_x, em,
                                     two_embed, nloc, nnei, last_layer_size);
} else if (device == "CPU") {
    deepmd::tabulate_fusion_se_a_cpu(descriptor, table, table_info, em_x, em,
                                     two_embed, nloc, nnei, last_layer_size);
}
```

## TensorFlow 后端实现差异

TensorFlow 后端的实现与 PyTorch 有所不同：

### 主要差异

1. **重新训练方式**: TensorFlow 后端通过重新训练模型来生成压缩版本
2. **图操作**: 使用 TensorFlow 的图模式进行优化
3. **静态图**: 生成的压缩模型是静态图格式

### 实现流程

```python
def compress(
    *,
    input: str,
    output: str,
    extrapolate: int,
    step: float,
    frequency: str,
    checkpoint_folder: str,
    training_script: str,
    # ... 其他参数
) -> None:
    # 1. 加载原始模型
    graph, _ = load_graph_def(input)

    # 2. 创建压缩配置文件
    jdata["model"]["compress"] = {}
    jdata["model"]["compress"]["model_file"] = input
    jdata["model"]["compress"]["table_config"] = [
        extrapolate, step, 10 * step, int(frequency)
    ]

    # 3. 重新训练模型
    train(...)

    # 4. 冻结模型
    freeze(checkpoint_folder=checkpoint_folder, output=output, node_names=None)
```

## 性能优化效果

### 加速效果

- **推理速度**: 通常可以达到 10 倍以上的加速
- **内存使用**: 可以减少 20 倍以上的内存使用
- **GPU 利用率**: 更好的内存访问模式提高 GPU 利用率

### 精度保持

- **相对误差**: 通常在 1%以内
- **力场一致性**: 保持物理量的正确性
- **能量守恒**: 维持系统能量守恒特性

## 使用建议

### 参数选择

1. **stride 参数**: 控制表格精度，越小精度越高但内存使用越大
2. **extrapolate 参数**: 控制外推范围，需要根据应用场景选择
3. **check_frequency**: 控制溢出检查频率，影响性能

### 适用场景

1. **生产环境**: 适合大规模 MD 模拟
2. **实时应用**: 适合需要快速响应的场景
3. **资源受限**: 适合内存和计算资源受限的环境

### 注意事项

1. **模型版本**: 确保使用支持压缩的模型版本
2. **描述符类型**: 确认描述符类型支持压缩
3. **训练数据**: 需要提供足够的训练数据来计算统计信息
4. **验证测试**: 建议在压缩后进行充分的验证测试

## 总结

DeepMD-kit 的 compress 功能通过表格化推理、算子融合和精确邻域索引三种技术实现了模型推理性能的显著提升。在保持模型精度的同时，可以获得 10 倍以上的性能提升和 20 倍以上的内存节省。这些技术不仅优化了计算效率，还提高了内存访问的局部性，是深度学习在科学计算领域应用的重要优化案例。

请详细梳理这个程序的 compress 功能，尤其是针对 pytorch 后端的，整个实现的代码执行流程，以及具体的实现方法，然后整理到 doc/outisli/compress.md 这个 markdown 文件中，一定要准确，详细，充分完全，逻辑条理清晰
