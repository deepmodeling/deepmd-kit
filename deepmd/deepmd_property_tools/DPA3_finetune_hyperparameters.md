# DPA3 预训练微调参数说明

本文说明使用 `DPA-3.2-5M.pt` 这类 DPA3 预训练模型做分子性质微调时，哪些参数应与预训练模型保持一致，哪些参数可以根据新任务自行设置。

## 1. 总体原则

预训练微调可以理解为：

```text
DPA3 descriptor 使用预训练模型权重
property fitting net / property head 面向新性质重新训练
```

因此参数可以分成两类：

```text
模型结构参数：应尽量和预训练模型一致，否则权重加载会失败
训练任务参数：可以按当前数据和任务重新设置
```

在当前 `deepmd_property_tools` 中，推荐使用：

```python
PropertyTrain(
    ...,
    finetune=PRETRAINED_MODEL,
    use_pretrain_script=True,
)
```

其中 `use_pretrain_script=True` 会让 DeePMD-kit 根据预训练模型里的 `model_params` 自动修正当前 `input.json` 中的模型结构，使其更容易和 `DPA-3.2-5M.pt` 对齐。

---

## 2. 应与预训练模型保持一致的参数

这些参数通常决定模型权重张量的形状或模型 forward 逻辑。如果和预训练模型不一致，容易出现：

```text
size mismatch
missing key
unexpected key
```

### 2.1 `model.type_map`

示例：

```json
"type_map": ["H", "C", "N", "O"]
```

微调数据中的元素类型应被预训练模型支持。当前 20 条 demo 数据自动生成：

```json
["H", "C", "N", "O"]
```

如果使用全量数据且包含 `I`，则可能生成：

```json
["H", "C", "N", "O", "I"]
```

需要确认预训练模型支持这些元素。

### 2.2 `model.descriptor.type`

必须是：

```json
"type": "dpa3"
```

因为微调目标是继承 DPA3 descriptor。

### 2.3 DPA3 repflow 维度参数

这些参数应与预训练模型一致：

```json
"n_dim": 128,
"e_dim": 64,
"a_dim": 32
```

含义：

- `n_dim`：节点表示维度
- `e_dim`：边表示维度
- `a_dim`：角表示维度

这些参数改变后，descriptor 内部权重矩阵形状会改变。

### 2.4 DPA3 层数

```json
"nlayers": 24
```

注意：当前工具原始 `input.json` 模板中可能是：

```json
"nlayers": 16
```

但使用 `DPA-3.2-5M.pt` 并开启 `use_pretrain_script=True` 后，DeePMD-kit 会在 `input_v2_compat.json` / `out.json` 中把它改成预训练模型实际使用的层数，例如：

```json
"nlayers": 24
```

这类结构参数应以预训练模型为准。

### 2.5 cutoff 和 neighbor selection 参数

这些参数建议和预训练模型一致：

```json
"e_rcut": 6.0,
"e_rcut_smth": 5.3,
"e_sel": 1200,
"a_rcut": 4.0,
"a_rcut_smth": 3.5,
"a_sel": 300,
"axis_neuron": 4
```

含义：

- `e_rcut` / `e_rcut_smth`：边距离 cutoff 与平滑区间
- `e_sel`：边邻居选择数量
- `a_rcut` / `a_rcut_smth`：角相关 cutoff 与平滑区间
- `a_sel`：角邻居选择数量
- `axis_neuron`：descriptor 内部投影维度相关参数

### 2.6 activation 和其他 descriptor 开关

预训练兼容后的配置中可能包含：

```json
"activation_function": "custom_silu:3.0",
"precision": "float32",
"use_tebd_bias": false,
"concat_output_tebd": false,
"use_loc_mapping": true,
"skip_stat": true,
"edge_init_use_dist": true,
"use_exp_switch": true,
"n_multi_edge_message": 1,
"optim_update": true
```

这些参数有些会影响模型结构，有些会影响模型计算逻辑。做预训练微调时，不建议手动随意修改。

---

## 3. 可以根据当前任务设置的参数

这些参数主要控制当前微调任务，不需要和预训练模型完全一致。

### 3.1 训练数据路径

例如：

```json
"training_data": {
  "systems": [
    "prepared_data/train/10",
    "prepared_data/train/15"
  ]
}
```

这些应使用当前任务生成的数据路径。

### 3.2 验证数据路径

例如：

```json
"validation_data": {
  "systems": [
    "prepared_data/valid/22"
  ]
}
```

同样由当前任务数据决定。

### 3.3 训练步数

可以自行设置：

```python
numb_steps=10
```

或正式训练时设置更大：

```python
numb_steps=10000
numb_steps=50000
numb_steps=200000
```

当前 20 条 demo 数据只用于 smoke test，`10` steps 只是验证流程。

### 3.4 batch size

可以根据数据量和显存调整：

```python
batch_size=1
```

或使用 DeePMD 支持的自动 batch：

```python
batch_size="auto:512"
```

当前 20 条 demo 数据中很多 system 只有 1-2 个样本，如果设置：

```python
batch_size=1024
```

会出现 warning：

```text
required batch size is larger than the size of the dataset
```

这不是致命错误，但小数据测试时 `batch_size=1` 更自然。

### 3.5 learning rate

微调通常使用比从头训练更小的学习率。

从头训练常见：

```json
"start_lr": 1e-3
```

预训练微调可用：

```json
"start_lr": 1e-4,
"stop_lr": 1e-6
```

在 `train_property_20.py` 中可通过 `input_updates` 设置：

```python
input_updates={
    "learning_rate": {
        "type": "exp",
        "decay_steps": 1000,
        "start_lr": 1e-4,
        "stop_lr": 1e-6,
    }
}
```

### 3.6 loss

性质预测任务使用：

```json
"loss": {
  "type": "property",
  "metric": ["mae", "rmse"],
  "loss_func": "smooth_mae",
  "beta": 1.0
}
```

这个由新任务决定，不需要和预训练模型原任务一致。

### 3.7 property name / property column

例如：

```python
property_name="Property"
property_col="Property"
```

含义：

- `property_col`：CSV 中读取哪一列作为标签
- `property_name`：写入 DeePMD 数据和 fitting net 的性质名

如果以后换性质，只需要对应修改这两个参数。

### 3.8 property fitting net

例如：

```json
"fitting_net": {
  "type": "property",
  "property_name": "Property",
  "intensive": true,
  "task_dim": 1,
  "neuron": [240, 240, 240]
}
```

对于新性质任务，fitting net 通常会重新初始化并训练。日志中出现：

```text
The fitting net will be re-init instead of using that in the pretrained model!
```

表示当前任务使用了新的 property head。

初期建议保持默认结构，确认流程稳定后再调 `neuron`、`task_dim` 等参数。

### 3.9 freeze

这是 `deepmd_property_tools` 的工具层参数：

```python
freeze=False
```

它控制训练结束后是否自动导出 `frozen_model.pth`。

当前 DPA3 预训练模型的 `custom_silu` 在 TorchScript freeze 阶段可能报错，因此当前 demo 中使用：

```python
freeze=False
```

先保存 checkpoint：

```text
model.ckpt-10.pt
```

并直接用 checkpoint 做预测。

### 3.10 `nproc_per_node`

这是 `deepmd_property_tools` 的训练启动参数，用于控制单节点启动多少个训练进程：

```python
nproc_per_node=1
```

默认值是 `1`，表示单进程训练。单进程时，工具会直接调用 DeePMD-kit 的 Python 训练入口。

如果设置为大于 1，例如：

```python
nproc_per_node=2
```

工具会改用 `torchrun` 启动多进程训练，等价于：

```bash
torchrun --nproc_per_node=2 --no-python dp --pt train input.json
```

通常含义是单节点 2 张 GPU / 2 个训练进程。8 卡训练可以设置：

```python
nproc_per_node=8
```

注意：`nproc_per_node` 不是 CPU 线程数。如果只是在 CPU 上想使用更多线程，应通过环境变量控制，例如：

```bash
export OMP_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=2
python train_property_20.py
```

---

## 4. 当前推荐配置示例

```python
trainer = PropertyTrain(
    task="regression",
    data_type="molecule",
    property_name="Property",
    property_col="Property",
    save_path=ROOT / "exp_property_20",
    numb_steps=10,
    batch_size=1024,
    model_name="dpa3",
    model_size="5m",
    freeze=False,
    nproc_per_node=1,
    finetune=ROOT / "DPA-3.2-5M.pt",
    use_pretrain_script=True,
    input_updates={
        "learning_rate": {
            "type": "exp",
            "decay_steps": 1000,
            "start_lr": 1e-4,
            "stop_lr": 1e-6,
        }
    },
)
```

对于更正式的训练，可以优先调整：

```text
numb_steps
batch_size
learning_rate
train_ratio
nproc_per_node
property_name / property_col
```

不建议优先手动修改：

```text
model.descriptor.repflow.*
activation_function
precision
DPA3 结构开关
```

这些应由 `use_pretrain_script=True` 自动继承预训练模型配置。

---

## 5. 简要总结

应继承预训练模型的主要是：

```text
DPA3 descriptor 结构参数
repflow 维度、层数、cutoff、sel
activation_function
precision
与 type_map 兼容的元素设置
```

可以自行设置的是：

```text
训练/验证数据
batch_size
numb_steps
learning_rate
loss
property_name / property_col
property fitting head
是否 freeze
nproc_per_node
```

当前工具推荐让 DeePMD-kit 通过：

```python
use_pretrain_script=True
```

自动继承预训练模型结构，而用户主要调当前任务相关的训练超参。
