# Learning rate

## Theory

The learning rate schedule consists of two phases: an optional warmup phase followed by a decay phase.

### Warmup phase (optional)

During the warmup phase (steps $0 \leq \tau < \tau^{\text{warmup}}$), the learning rate increases linearly from an initial warmup learning rate to the target starting learning rate:

```math
    \gamma(\tau) = \gamma^{\text{warmup}} + \frac{\gamma^0 - \gamma^{\text{warmup}}}{\tau^{\text{warmup}}} \tau,
```

where $\gamma^{\text{warmup}} = f^{\text{warmup}} \cdot \gamma^0$ is the initial warmup learning rate, $f^{\text{warmup}} \in [0, 1]$ is the warmup start factor (default 0.0), and $\tau^{\text{warmup}} \in \mathbb{N}$ is the number of warmup steps.

### Decay phase

After the warmup phase (steps $\tau \geq \tau^{\text{warmup}}$), the learning rate decays according to the selected schedule type.

**Exponential decay (`type: "exp"`):**

The learning rate decays exponentially:

```math
    \gamma(\tau) = \gamma^0 r ^ {\lfloor  (\tau - \tau^{\text{warmup}})/s \rfloor},
```

where $\tau \in \mathbb{N}$ is the index of the training step, $\gamma^0  \in \mathbb{R}$ is the learning rate at the start of the decay phase (i.e., after warmup), and the decay rate $r$ is given by

```math
    r = {\left(\frac{\gamma^{\text{stop}}}{\gamma^0}\right )} ^{\frac{s}{\tau^{\text{decay}}}},
```

where $\tau^{\text{decay}} = \tau^{\text{stop}} - \tau^{\text{warmup}}$ is the number of decay steps, $\tau^{\text{stop}} \in \mathbb{N}$ is the total training steps, $\gamma^{\text{stop}} \in \mathbb{R}$ is the stopping learning rate, and $s \in \mathbb{N}$ is the decay steps.

**Cosine annealing (`type: "cosine"`):**

The learning rate follows a cosine annealing schedule:

```math
    \gamma(\tau) = \gamma^{\text{stop}} + \frac{\gamma^0 - \gamma^{\text{stop}}}{2} \left(1 + \cos\left(\frac{\pi (\tau - \tau^{\text{warmup}})}{\tau^{\text{decay}}}\right)\right),
```

where the learning rate smoothly decreases from $\gamma^0$ to $\gamma^{\text{stop}}$ following a cosine curve over the decay phase.

For both schedule types, the stopping learning rate can be specified directly as $\gamma^{\text{stop}}$ or as a ratio: $\gamma^{\text{stop}} = \rho^{\text{stop}} \cdot \gamma^0$, where $\rho^{\text{stop}} \in (0, 1]$ is the stopping learning rate ratio.
[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Migration Guide

### Required parameters for learning rate configuration

Starting from this version (3.1.3), the learning rate configuration has the following **required** parameters:

1. **`start_lr`** (required): The learning rate at the start of the decay phase (after warmup). This parameter no longer has a default value and must be explicitly specified in your configuration.

2. **Either `stop_lr` or `stop_lr_ratio`** (required): You must provide one of these two parameters:
   - `stop_lr`: The target learning rate at the end of training
   - `stop_lr_ratio`: The stopping learning rate as a ratio of `start_lr`

These parameters are mutually exclusive - you cannot specify both `stop_lr` and `stop_lr_ratio` at the same time.

#### Migration examples

**Before (legacy configuration):**

```json
"learning_rate": {
    "type": "exp",
    "decay_steps": 5000
}
```

**After (updated configuration):**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000
}
```

Or using `stop_lr_ratio`:

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3,
    "decay_steps": 5000
}
```

**Note:** If you are upgrading from a previous version, please update your configuration files to include explicit values for `start_lr` and one of `stop_lr` or `stop_lr_ratio`. Failure to do so will result in a validation error.

## Instructions

DeePMD-kit supports two types of learning rate schedules: exponential decay (`type: "exp"`) and cosine annealing (`type: "cosine"`). Both types support optional warmup and can use either absolute stopping learning rate or a ratio-based specification.

### Exponential decay schedule

The {ref}`learning_rate <learning_rate>` section for exponential decay in `input.json` is given as follows

```json
    "learning_rate" :{
        "type":        "exp",
        "start_lr":    0.001,
        "stop_lr":     1e-6,
        "decay_steps": 5000,
        "_comment":    "that's all"
    }
```

#### Basic parameters

The following parameters are available for learning rate configuration.

**Common parameters for both `exp` and `cosine` types:**

- {ref}`start_lr <learning_rate[exp]/start_lr>` gives the learning rate at the start of the decay phase (i.e., after warmup if enabled). It should be set appropriately based on the model architecture and dataset.
- {ref}`stop_lr <learning_rate[exp]/stop_lr>` gives the target learning rate at the end of the training. It should be small enough to ensure that the network parameters satisfactorily converge. This parameter is mutually exclusive with {ref}`stop_lr_ratio <learning_rate[exp]/stop_lr_ratio>`.
- {ref}`stop_lr_ratio <learning_rate[exp]/stop_lr_ratio>` (optional) specifies the stopping learning rate as a ratio of {ref}`start_lr <learning_rate[exp]/start_lr>`. For example, `stop_lr_ratio: 1e-3` means `stop_lr = start_lr * 1e-3`. This parameter is mutually exclusive with {ref}`stop_lr <learning_rate[exp]/stop_lr>`. Either {ref}`stop_lr <learning_rate[exp]/stop_lr>` or {ref}`stop_lr_ratio <learning_rate[exp]/stop_lr_ratio>` must be provided.

**Additional parameters for `exp` type only:**

- {ref}`decay_steps <learning_rate[exp]/decay_steps>` specifies the interval (in training steps) at which the learning rate is decayed. The learning rate is updated every {ref}`decay_steps <learning_rate[exp]/decay_steps>` steps during the decay phase. If `decay_steps` exceeds the decay phase steps (num_steps - warmup_steps) and `decay_rate` is not explicitly provided, it will be automatically adjusted to a sensible default value.
- {ref}`smooth <learning_rate[exp]/smooth>` (optional, default: `false`) controls the decay behavior. When set to `false`, the learning rate decays in a stepped manner (updated every `decay_steps` steps). When set to `true`, the learning rate decays smoothly at every step.

**Learning rate formula for `exp` type:**

During the decay phase, the learning rate decays exponentially from {ref}`start_lr <learning_rate[exp]/start_lr>` to {ref}`stop_lr <learning_rate[exp]/stop_lr>`.

- **Stepped mode (`smooth: false`, default):**

```text
lr(t) = start_lr * decay_rate ^ floor((t - warmup_steps) / decay_steps)
```

- **Smooth mode (`smooth: true`):**

```text
lr(t) = start_lr * decay_rate ^ ((t - warmup_steps) / decay_steps)
```

where `t` is the current training step and `warmup_steps` is the number of warmup steps (0 if warmup is not enabled).

The formula for cosine annealing is as follows.

**Learning rate formula for `cosine` type:**

For cosine annealing, the learning rate smoothly decreases following a cosine curve:

```text
lr(t) = stop_lr + (start_lr - stop_lr) / 2 * (1 + cos(pi * (t - warmup_steps) / decay_phase_steps))
```

where `decay_phase_steps = numb_steps - warmup_steps` is the number of steps in the decay phase.

#### Warmup parameters (optional)

Warmup is a technique to stabilize training in the early stages by gradually increasing the learning rate from a small initial value to the target {ref}`start_lr <learning_rate[exp]/start_lr>`. The warmup parameters are optional and can be configured as follows:

- {ref}`warmup_steps <learning_rate[exp]/warmup_steps>` (optional, default: 0) specifies the number of steps for learning rate warmup. During warmup, the learning rate increases linearly from `warmup_start_factor * start_lr` to {ref}`start_lr <learning_rate[exp]/start_lr>`. This parameter is mutually exclusive with {ref}`warmup_ratio <learning_rate[exp]/warmup_ratio>`.
- {ref}`warmup_ratio <learning_rate[exp]/warmup_ratio>` (optional) specifies the warmup duration as a ratio of the total training steps. For example, `warmup_ratio: 0.1` means the warmup phase will last for 10% of the total training steps. The actual number of warmup steps is computed as `int(warmup_ratio * numb_steps)`. This parameter is mutually exclusive with {ref}`warmup_steps <learning_rate[exp]/warmup_steps>`.
- {ref}`warmup_start_factor <learning_rate[exp]/warmup_start_factor>` (optional, default: 0.0) specifies the factor for the initial warmup learning rate. The warmup learning rate starts from `warmup_start_factor * start_lr` and increases linearly to {ref}`start_lr <learning_rate[exp]/start_lr>`. A value of 0.0 means the learning rate starts from zero.

#### Configuration examples

The following examples demonstrate various learning rate configurations.

**Example 1: Basic exponential decay without warmup**

```json
    "learning_rate": {
        "type":        "exp",
        "start_lr":    0.001,
        "stop_lr":     1e-6,
        "decay_steps": 5000
    }
```

**Example 2: Using stop_lr_ratio instead of stop_lr**

```json
    "learning_rate": {
        "type":          "exp",
        "start_lr":      0.001,
        "stop_lr_ratio": 1e-3,
        "decay_steps":   5000
    }
```

This is equivalent to setting `stop_lr: 1e-6` (i.e., `0.001 * 1e-3`).

The following example shows exponential decay with warmup using a specific number of warmup steps.

**Example 3: Exponential decay with warmup (using warmup_steps)**

```json
    "learning_rate": {
        "type":               "exp",
        "start_lr":           0.001,
        "stop_lr":            1e-6,
        "decay_steps":        5000,
        "warmup_steps":       10000,
        "warmup_start_factor": 0.1
    }
```

In this example, the learning rate starts from `0.0001` (i.e., `0.1 * 0.001`) and increases linearly to `0.001` over the first 10,000 steps. After that, it decays exponentially to `1e-6`.

The following example shows exponential decay with warmup using a ratio-based warmup duration.

**Example 4: Exponential decay with warmup (using warmup_ratio)**

```json
    "learning_rate": {
        "type":          "exp",
        "start_lr":      0.001,
        "stop_lr_ratio": 1e-3,
        "decay_steps":   5000,
        "warmup_ratio":  0.05
    }
```

In this example, if the total training steps (`numb_steps`) is 1,000,000, the warmup phase will last for 50,000 steps (i.e., `0.05 * 1,000,000`). The learning rate starts from `0.0` (default `warmup_start_factor: 0.0`) and increases linearly to `0.001` over the first 50,000 steps, then decays exponentially.

The following examples demonstrate cosine annealing configurations.

### Cosine annealing schedule

The {ref}`learning_rate <learning_rate>` section for cosine annealing in `input.json` is given as follows

```json
    "learning_rate": {
        "type":     "cosine",
        "start_lr": 0.001,
        "stop_lr":  1e-6
    }
```

Cosine annealing provides a smooth decay curve that often works well for training neural networks. Unlike exponential decay, it does not require the `decay_steps` parameter.

The following example shows basic cosine annealing without warmup.

**Example 5: Basic cosine annealing without warmup**

```json
    "learning_rate": {
        "type":     "cosine",
        "start_lr": 0.001,
        "stop_lr":  1e-6
    }
```

The following example shows cosine annealing with stop_lr_ratio.

**Example 6: Cosine annealing with stop_lr_ratio**

```json
    "learning_rate": {
        "type":          "cosine",
        "start_lr":      0.001,
        "stop_lr_ratio": 1e-3
    }
```

This is equivalent to setting `stop_lr: 1e-6` (i.e., `0.001 * 1e-3`).

The following example shows cosine annealing with warmup.

**Example 7: Cosine annealing with warmup**

```json
    "learning_rate": {
        "type":               "cosine",
        "start_lr":           0.001,
        "stop_lr":            1e-6,
        "warmup_steps":       5000,
        "warmup_start_factor": 0.0
    }
```

In this example, the learning rate starts from `0.0` and increases linearly to `0.001` over the first 5,000 steps, then follows a cosine annealing curve down to `1e-6`.

The following example shows exponential decay with smooth mode enabled.

**Example 8: Exponential decay with smooth mode**

```json
    "learning_rate": {
        "type":        "exp",
        "start_lr":    0.001,
        "stop_lr":     1e-6,
        "decay_steps": 5000,
        "smooth":      true
    }
```

By setting `smooth: true`, the learning rate decays smoothly at every step instead of in a stepped manner. This provides a more gradual decay curve similar to PyTorch's `ExponentialLR`, whereas the default stepped mode (`smooth: false`) is similar to PyTorch's `StepLR`.
