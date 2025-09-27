# DeePMD-kit 压缩功能详细分析

## 概述

DeePMD-kit 的 compress 功能通过将 embedding networks 进行 tabulation（查表法）来实现模型压缩，显著提升推理速度并减少内存占用。

## 核心原理

### 基本思想

1. **预计算查表**：将 embedding networks 的输出预先计算并存储在表格中
2. **分段插值**：使用两个不同步长的表格来平衡精度与存储成本：
   - 第一段表格：使用精细步长（stride0）
   - 第二段表格：使用粗糙步长（stride1 = 10 × stride0）
3. **多项式插值**：基于查表结果进行五次多项式插值

## PyTorch 后端实现

### 1. 命令行入口

#### 主入口
- **文件位置**: `deepmd/main.py`
- **命令示例**: `dp --pt compress -i model.pth -o compressed_model.pth`

#### 参数配置
```python
parser_compress.add_argument("-s", "--step", default=0.01, type=float)      # stride0
parser_compress.add_argument("-e", "--extrapolate", default=5, type=int)    # 外推倍数
parser_compress.add_argument("-f", "--frequency", default=-1, type=int)     # 溢出检查频率
parser_compress.add_argument("-t", "--training-script", type=str)           # 训练脚本
```

#### 命令分发
```python
# deepmd/main.py:1013-1018
elif args.command in ("compress", "train", "freeze", ...):
    deepmd_main = BACKENDS[args.backend]().entry_point_hook
```

### 2. PyTorch 后端处理

#### 入口函数
**文件位置**: `deepmd/pt/entrypoints/main.py:574-582`

```python
elif FLAGS.command == "compress":
    FLAGS.input = str(Path(FLAGS.input).with_suffix(".pth"))
    FLAGS.output = str(Path(FLAGS.output).with_suffix(".pth"))
    enable_compression(
        input_file=FLAGS.input,
        output=FLAGS.output,
        stride=FLAGS.step,
        extrapolate=FLAGS.extrapolate,
        check_frequency=FLAGS.frequency,
        training_script=FLAGS.training_script,
    )
```

#### 核心压缩函数
**文件位置**: `deepmd/pt/entrypoints/compress.py:32-84`

## 详细执行流程

### 步骤1：模型加载

```python
def enable_compression(input_file, output, stride=0.01, extrapolate=5, check_frequency=-1, training_script=None):
    # 1. 加载JIT模型
    saved_model = torch.jit.load(input_file, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    
    # 2. 重建模型实例
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict())
```

### 步骤2：最小邻居距离计算

```python
# 3. 计算最小邻居距离
if model.get_min_nbor_dist() is None:
    # 从训练数据计算
    jdata = j_loader(training_script)
    jdata = update_deepmd_input(jdata)
    train_data = get_data(jdata["training"]["training_data"], 0, type_map, None)
    
    update_sel = UpdateSel()
    t_min_nbor_dist = update_sel.get_min_nbor_dist(train_data)
    model.min_nbor_dist = torch.tensor(t_min_nbor_dist, dtype=env.GLOBAL_PT_FLOAT_PRECISION)
```

### 步骤3：模型压缩启用

#### 3.1 模型层压缩
**文件位置**: `deepmd/pt/model/model/make_model.py:103-129`

```python
def enable_compression(self, table_extrapolate=5, table_stride_1=0.01, table_stride_2=0.1, check_frequency=-1):
    """模型层压缩入口"""
    self.atomic_model.enable_compression(
        self.get_min_nbor_dist(),  # 最小邻居距离
        table_extrapolate,
        table_stride_1,
        table_stride_2, 
        check_frequency,
    )
```

#### 3.2 原子模型压缩
**文件位置**: `deepmd/pt/model/atomic_model/dp_atomic_model.py:188-217`

```python
def enable_compression(self, min_nbor_dist, table_extrapolate=5, table_stride_1=0.01, table_stride_2=0.1, check_frequency=-1):
    """原子模型层压缩入口"""
    self.descriptor.enable_compression(
        min_nbor_dist,
        table_extrapolate,
        table_stride_1,
        table_stride_2,
        check_frequency,
    )
```

### 步骤4：描述符层压缩实现

#### 4.1 SE_A 描述符压缩
**文件位置**: `deepmd/pt/model/descriptor/se_a.py:257-302`

```python
def enable_compression(self, min_nbor_dist, table_extrapolate=5, table_stride_1=0.01, table_stride_2=0.1, check_frequency=-1):
    # 1. 检查是否已压缩
    if self.compress:
        raise ValueError("Compression is already enabled.")
    
    # 2. 创建查表器
    data = self.serialize()
    self.table = DPTabulate(
        self,                                    # 描述符对象
        data["neuron"],                          # 神经网络结构
        data["type_one_side"],                   # 单侧类型
        data["exclude_types"],                   # 排除类型对
        ActivationFn(data["activation_function"]) # 激活函数
    )
    
    # 3. 存储查表配置
    self.table_config = [table_extrapolate, table_stride_1, table_stride_2, check_frequency]
    
    # 4. 构建查表数据
    self.lower, self.upper = self.table.build(min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2)
    
    # 5. 启用嵌入层压缩
    self.sea.enable_compression(self.table.data, self.table_config, self.lower, self.upper)
    
    # 6. 设置压缩标志
    self.compress = True
```

#### 4.2 DescrptSeA 压缩数据设置
**文件位置**: `deepmd/pt/model/descriptor/se_a.py:699-733`

```python
def enable_compression(self, table_data, table_config, lower, upper):
    """为每个嵌入网络设置压缩数据"""
    for embedding_idx, ll in enumerate(self.filter_layers.networks):
        if self.type_one_side:
            net = f"filter_-1_net_{embedding_idx}"
        else:
            ii = embedding_idx // self.ntypes  # 中心原子类型
            ti = embedding_idx % self.ntypes   # 邻居原子类型  
            net = f"filter_{ii}_net_{ti}"
            
        # 压缩信息：[lower, upper, upper*extrapolate, stride1, stride2, check_freq]
        info_ii = torch.as_tensor([
            lower[net], upper[net], upper[net] * table_config[0],
            table_config[1], table_config[2], table_config[3]
        ], dtype=self.prec, device="cpu")
        
        # 压缩数据：多项式系数表
        tensor_data_ii = table_data[net].to(device=env.DEVICE, dtype=self.prec)
        
        self.compress_data[embedding_idx] = tensor_data_ii
        self.compress_info[embedding_idx] = info_ii
    
    self.compress = True
```

### 步骤5：查表器实现

#### 5.1 查表器类
**文件位置**: `deepmd/pt/utils/tabulate.py:52-118`

```python
class DPTabulate(BaseTabulate):
    def __init__(self, descrpt, neuron, type_one_side=False, exclude_types=[], activation_fn=ActivationFn("tanh")):
        # 1. 基础初始化
        super().__init__(descrpt, neuron, type_one_side, exclude_types, True)
        
        # 2. 描述符类型判断
        self.descrpt_type = self._get_descrpt_type()  # "A", "Atten", "T", "R"
        
        # 3. 获取描述符参数
        self.sel_a = self.descrpt.get_sel()
        self.rcut = self.descrpt.get_rcut()
        self.rcut_smth = self.descrpt.get_rcut_smth()
        
        # 4. 激活函数映射
        activation_map = {"tanh": 1, "gelu": 2, "relu": 3, "relu6": 4, "softplus": 5, "sigmoid": 6}
        self.functype = activation_map[activation_fn.activation]
        
        # 5. 获取统计参数
        serialized = self.descrpt.serialize()
        self.davg = serialized["@variables"]["davg"]  # 均值
        self.dstd = serialized["@variables"]["dstd"]  # 标准差
        self.embedding_net_nodes = serialized["embeddings"]["networks"]
        
        # 6. 提取权重和偏置
        self.bias = self._get_bias()
        self.matrix = self._get_matrix()
```

#### 5.2 查表构建过程
**文件位置**: `deepmd/utils/tabulate.py:70-243`

```python
def build(self, min_nbor_dist, extrapolate, stride0, stride1):
    # 1. 计算环境矩阵范围
    lower, upper = self._get_env_mat_range(min_nbor_dist)
    
    # 2. 根据描述符类型建表
    if self.descrpt_type == "A":  # SE_A 描述符
        for ii in range(self.table_size):
            if self._should_build_table(ii):
                # 构建距离网格
                xx = self._build_distance_grid(lower, upper, stride0, stride1, extrapolate, ii)
                
                # 查表数据
                self._generate_spline_table(net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline)
    
    # 3. 后处理转换
    self._convert_numpy_to_tensor()
    self._convert_numpy_float_to_int()
    
    return self.lower, self.upper
```

#### 5.3 环境矩阵范围计算
**文件位置**: `deepmd/utils/tabulate.py:445-463`

```python
def _get_env_mat_range(self, min_nbor_dist):
    """计算环境矩阵的范围"""
    # 1. 计算切换函数值
    sw = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
    
    # 2. 根据描述符类型计算范围
    if self.descrpt_type in ("Atten", "A"):
        # 标准化：(r_ij - davg) / dstd
        lower = -self.davg[:, 0] / self.dstd[:, 0]
        upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
    
    # 3. 向下和向上取整
    return np.floor(lower), np.ceil(upper)
```

#### 5.4 多项式系数计算
**文件位置**: `deepmd/utils/tabulate.py:245-347`

```python
def _generate_spline_table(self, net, xx, idx, upper, lower, stride0, stride1, extrapolate, nspline):
    # 1. 通过神经网络前向传播计算数据
    vv, dd, d2 = self._make_data(xx, idx)  # 值、一阶导数、二阶导数
    
    # 2. 多项式系数表
    self.data[net] = np.zeros([nspline, 6 * self.last_layer_size], dtype=self.data_type)
    
    # 3. 步长处理
    tt = np.full((nspline, self.last_layer_size), stride1)
    tt[: int((upper - lower) / stride0), :] = stride0
    
    # 4. 计算多项式高阶系数
    hh = vv[1:nspline + 1, :self.last_layer_size] - vv[:nspline, :self.last_layer_size]
    
    # 系数0：函数值 f(x)
    self.data[net][:, ::6] = vv[:nspline, :self.last_layer_size]
    
    # 系数1：一阶导数 f'(x)
    self.data[net][:, 1::6] = dd[:nspline, :self.last_layer_size]
    
    # 系数2：二阶导数 f''(x)/2
    self.data[net][:, 2::6] = 0.5 * d2[:nspline, :self.last_layer_size]
    
    # 系数3-5：高阶多项式系数（保证连续性）
    self.data[net][:, 3::6] = (1 / (2 * tt**3)) * (20 * hh - ...)
    self.data[net][:, 4::6] = (1 / (2 * tt**4)) * (-30 * hh + ...)
    self.data[net][:, 5::6] = (1 / (2 * tt**5)) * (12 * hh - ...)
```

#### 5.5 神经网络前向传播
**文件位置**: `deepmd/pt/utils/tabulate.py:119-250`

```python
def _make_data(self, xx, idx):
    """通过神经网络前向传播查表数据"""
    xx = torch.from_numpy(xx).view(-1, 1).to(env.DEVICE)
    
    # 逐层计算
    for layer in range(self.layer_size):
        if layer == 0:
            # 第一层：线性变换 + 激活函数
            xbar = torch.matmul(xx, torch.from_numpy(self.matrix[f"layer_{layer + 1}"][idx])) + \
                   torch.from_numpy(self.bias[f"layer_{layer + 1}"][idx])
            
            # 处理激活函数（含残差连接）
            if self.neuron[0] == 1:
                yy = self._layer_0(...) + xx  # 残差连接
            else:
                yy = self._layer_0(...)
            
            # 计算一阶和二阶导数
            dy = unaggregated_dy_dx_s(...)
            dy2 = unaggregated_dy2_dx_s(...)
        else:
            # 后续层...
    
    return vv.cpu().numpy(), dd.cpu().numpy(), d2.cpu().numpy()
```

### 步骤6：模型保存

```python
# 4. 启用压缩
model.enable_compression(extrapolate, stride, stride * 10, check_frequency)

# 5. JIT脚本化保存
model = torch.jit.script(model)
torch.jit.save(model, output)
```

## 支持的描述符类型

### 已支持的描述符

1. **SE_A (Smooth Edition Angular)**
   - **文件位置**: `deepmd/pt/model/descriptor/se_a.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 支持角度信息的描述符

2. **SE_R (Smooth Edition Radial)**
   - **文件位置**: `deepmd/pt/model/descriptor/se_r.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 仅使用径向距离信息的描述符

3. **SE_T (Smooth Edition Three-body)**
   - **文件位置**: `deepmd/pt/model/descriptor/se_t.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 三体相互作用描述符

4. **SE_Atten (Smooth Edition with Attention)**
   - **文件位置**: `deepmd/pt/model/descriptor/se_atten.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 带注意力机制的描述符

5. **DPA1 (Deep Potential Attention 1)**
   - **文件位置**: `deepmd/pt/model/descriptor/dpa1.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 第一代注意力机制描述符

6. **DPA2 (Deep Potential Attention 2)**
   - **文件位置**: `deepmd/pt/model/descriptor/dpa2.py`
   - **压缩方式**: 嵌入网络表格化
   - **特点**: 第二代注意力机制描述符

### 不支持的描述符

1. **DPA3 (Deep Potential Attention 3)**
   - **文件位置**: `deepmd/pt/model/descriptor/dpa3.py:578-601`
   - **压缩方式**: 不支持
   - **原因**: ```python
     def enable_compression(self, ...):
         raise NotImplementedError("Compression is unsupported for DPA3.")
     ```

### 特殊模型类型

1. **Linear Atomic Model**
   - **文件位置**: `deepmd/pt/model/atomic_model/linear_atomic_model.py:198-228`
   - **压缩方式**: 多个子模型分别压缩

2. **Pairtab Atomic Model**
   - **文件位置**: `deepmd/pt/model/atomic_model/pairtab_atomic_model.py:505-514`
   - **压缩方式**: 不支持查表压缩

## 数据结构详解

### 压缩数据格式

#### 1. 压缩信息 (compress_info)
```python
# 每个嵌入网络存储6个参数 [6]
compress_info[embedding_idx] = torch.tensor([
    lower[net],           # 下界
    upper[net],           # 上界  
    upper[net] * extrapolate,  # 外推上界
    table_stride_1,       # 第一段步长
    table_stride_2,       # 第二段步长  
    check_frequency       # 溢出检查频率
])
```

#### 2. 压缩数据 (compress_data)
```python
# 每个嵌入网络存储系数表 [nspline, 6 * last_layer_size]
compress_data[embedding_idx] = table_data[net]

# 其中每6个连续的系数表示一个多项式的系数
# [f(x), f'(x), f''(x)/2, c3, c4, c5] × last_layer_size
```

### 查表数据构建

#### 1. 距离网格生成
```python
# 第一段：精细数据区间网格
xx1 = np.arange(lower, upper, stride0)

# 第二段：外推区间网格  
xx2 = np.arange(upper, extrapolate * upper, stride1)

# 合并网格
xx = np.concatenate([xx1, xx2, [extrapolate * upper]])
```

#### 2. 神经网络求值
```python
# 对每个网格点进行神经网络前向传播
for x_point in xx:
    output = forward_pass(x_point)      # 网络输出
    grad1 = compute_gradient(x_point)   # 一阶导数
    grad2 = compute_hessian(x_point)    # 二阶导数
```

#### 3. 多项式构造
采用五次Hermite插值，满足：
- 函数值连续：f(x_i) = y_i
- 一阶导数连续：f'(x_i) = y'_i  
- 二阶导数连续：f''(x_i) = y''_i

## 性能优化

### 1. 内存管理
- **数据精度**: 支持数据精度调整（0.01）
- **分段优化**: 粗糙步长在外推区（0.1）
- **内存复用**: 删除原始网络权重，内存显著降低

### 2. 计算优化
- **预计算查表**: 压缩后嵌入网络不再需要矩阵运算
- **向量化查表**: 每个原子类型对应一个优化的查表
- **分支消除**: 消除类型判断的分支开销

### 3. 缓存友好
- **数据局部性**: 查表数据连续存储，提升cache命中率
- **内存访问**: 内存访问模式优化，减少cache miss
- **SIMD**: 多项式计算可向量化

## 使用示例

### 基础压缩命令
```bash
# 压缩PyTorch模型
dp --pt compress -i frozen_model.pth -o compressed_model.pth

# 自定义参数
dp --pt compress \
  -i frozen_model.pth \
  -o compressed_model.pth \
  -s 0.005 \
  -e 10 \
  -f 1000 \
  -t input.json
```

### 参数说明
- `-i, --input`: 输入的冻结模型（.pth）
- `-o, --output`: 输出的压缩模型（.pth）
- `-s, --step`: 第一段步长，影响精度与内存（默认0.01）
- `-e, --extrapolate`: 外推倍数（默认5）
- `-f, --frequency`: 溢出检查频率，-1表示不检查（默认-1）
- `-t, --training-script`: 训练脚本（用于计算最小邻居距离）

## 局限性分析

### 1. 描述符局限
- DPA3 描述符不支持压缩
- Pairtab 模型不支持查表压缩
- 某些描述符变体可能不完全兼容

### 2. 精度权衡
- 步长设置过大会影响精度
- 外推区间精度相对较低
- 激活函数近似可能带来误差

### 3. 内存开销
- 压缩后仍需存储多项式查表数据
- 精度要求高时查表尺寸增大
- 激活函数导数计算消耗额外内存

### 4. 兼容性限制
- 压缩后的模型仅适用于DeePMD-kit环境
- JIT脚本化可能在某些场景下受限
- LAMMPS等MD引擎需要特定的压缩模型格式

## 实现细节

### 多项式插值公式

在区间 [x_i, x_{i+1}] 内，对于变量 x，多项式为：

```
f(x) = c₀ + c₁t + c₂t² + c₃t³ + c₄t⁴ + c₅t⁵
```

其中：
- `t = (x - x_i) / h`，h 为步长
- `c₀ = f(x_i)`
- `c₁ = f'(x_i) × h`
- `c₂ = f''(x_i) × h² / 2`
- `c₃, c₄, c₅` 根据边界连续性确定

### 切换函数

用于平滑处理截断半径的切换函数：

```python
def spline5_switch(r, r_min, r_max):
    if r < r_min:
        return 1.0
    elif r < r_max:
        u = (r - r_min) / (r_max - r_min)
        return u³(-6u² + 15u - 10) + 1
    else:
        return 0.0
```

## 总结

DeePMD-kit的compress功能通过将神经网络嵌入层用查表法和多项式插值替代，实现了显著的推理加速。PyTorch后端的实现采用了分层设计，由模型层、原子模型层、描述符层逐级传递压缩请求。查表器构建了精细和粗糙分段的插值表，平衡了精度与性能。该功能对大多数SE类和DPA1/DPA2描述符提供良好支持，是生产环境中提升MD模拟效率的重要工具。