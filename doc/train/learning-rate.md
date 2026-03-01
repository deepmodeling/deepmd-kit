# Learning rate

DeePMD-kit supports two learning rate schedules:

- **`exp`**: Exponential decay with optional stepped or smooth mode
- **`cosine`**: Cosine annealing for smooth decay curve

Both schedules support an optional warmup phase where the learning rate gradually increases from a small initial value to the target `start_lr`.

## Quick Start

### Exponential decay (default)

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000
}
```

### Cosine annealing

```json
"learning_rate": {
    "type": "cosine",
    "start_lr": 0.001,
    "stop_lr": 1e-6
}
```

## Common parameters

The following parameters are shared by both `exp` and `cosine` schedules.

### Required parameters

- `start_lr`: The learning rate at the start of training (after warmup).
- `stop_lr` or `stop_lr_ratio` (must provide exactly one):
  - `stop_lr`: The learning rate at the end of training.
  - `stop_lr_ratio`: The ratio of `stop_lr` to `start_lr`. Computed as `stop_lr = start_lr * stop_lr_ratio`.

### Optional parameters

- `warmup_steps` or `warmup_ratio` (mutually exclusive):
  - `warmup_steps`: Number of steps for warmup. Learning rate increases linearly from `warmup_start_factor * start_lr` to `start_lr`.
  - `warmup_ratio`: Ratio of warmup steps to total training steps. `warmup_steps = int(warmup_ratio * numb_steps)`.
- `warmup_start_factor`: Factor for initial warmup learning rate (default: 0.0). Warmup starts from `warmup_start_factor * start_lr`.
- `scale_by_worker`: How to alter learning rate in parallel training. Options: `"linear"`, `"sqrt"`, `"none"` (default: `"linear"`).

### Type-specific parameters

**Exponential decay (`type: "exp"`):**

- `decay_steps`: Interval (in steps) at which learning rate decays (default: 5000).
- `decay_rate`: Explicit decay rate. If not provided, computed from `start_lr` and `stop_lr`.
- `smooth`: If `true`, use smooth exponential decay at every step. If `false`, use stepped decay (default: `false`).

**Cosine annealing (`type: "cosine"`):**

No type-specific parameters. The decay follows a cosine curve from `start_lr` to `stop_lr`.

See [Mathematical Theory](#mathematical-theory) section for complete formulas.

## Exponential Decay Schedule

The exponential decay schedule reduces the learning rate exponentially over training steps. It is the default schedule when `type` is omitted.

### Stepped vs smooth mode

By setting `smooth: true`, the learning rate decays smoothly at every step instead of in a stepped manner. This provides a more gradual decay curve similar to PyTorch's `ExponentialLR`, whereas the default stepped mode (`smooth: false`) is similar to PyTorch's `StepLR`.

### Decay rate computation

If `decay_rate` is not explicitly provided, it is computed from `start_lr` and `stop_lr` to ensure the learning rate reaches `stop_lr` at the end of training:

```text
decay_rate = (stop_lr / start_lr) ^ (decay_steps / (numb_steps - warmup_steps))
```

where `numb_steps` is the internal total number of training steps (derived from `training.numb_steps` in the training configuration).

### Examples

**Basic exponential decay without warmup:**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000
}
```

**Using `stop_lr_ratio`:**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3,
    "decay_steps": 5000
}
```

Equivalent to `stop_lr: 1e-6` (i.e., `0.001 * 1e-3`).

**With warmup (using `warmup_steps`):**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000,
    "warmup_steps": 10000,
    "warmup_start_factor": 0.1
}
```

Learning rate starts from `0.0001` (i.e., `0.1 * 0.001`), increases linearly to `0.001` over 10,000 steps, then decays exponentially.

**With warmup (using `warmup_ratio`):**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3,
    "decay_steps": 5000,
    "warmup_ratio": 0.05
}
```

If `numb_steps` is 1,000,000, warmup lasts 50,000 steps. Learning rate starts from `0.0` (default `warmup_start_factor`) and increases to `0.001`.

**Smooth exponential decay:**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000,
    "smooth": true
}
```

With `smooth: true`, the learning rate decays continuously at every step, similar to PyTorch's `ExponentialLR`. The default stepped mode (`smooth: false`) is similar to PyTorch's `StepLR`.

## Cosine Annealing Schedule

The cosine annealing schedule smoothly decreases the learning rate following a cosine curve. It often provides better convergence than exponential decay.

### Formula

During the decay phase (after warmup), the learning rate follows:

```text
lr(t) = stop_lr + (start_lr - stop_lr) / 2 * (1 + cos(pi * (t - warmup_steps) / (numb_steps - warmup_steps)))
```

At the middle of training (relative to decay phase), the learning rate is approximately `(start_lr + stop_lr) / 2`.

### Examples

**Basic cosine annealing:**

```json
"learning_rate": {
    "type": "cosine",
    "start_lr": 0.001,
    "stop_lr": 1e-6
}
```

**Using `stop_lr_ratio`:**

```json
"learning_rate": {
    "type": "cosine",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3
}
```

**With warmup:**

```json
"learning_rate": {
    "type": "cosine",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "warmup_steps": 5000,
    "warmup_start_factor": 0.0
}
```

## Warmup Mechanism

Warmup is a technique to stabilize training in early stages by gradually increasing the learning rate from a small initial value.

### Warmup formula

During warmup phase ($0 \leq \tau < \tau^{\text{warmup}}$):

```math
\gamma(\tau) = \gamma^{\text{warmup}} + (\gamma^0 - \gamma^{\text{warmup}}) \cdot \frac{\tau}{\tau^{\text{warmup}}}
```

where:

- $\tau$ is the current step index
- $\tau^{\text{warmup}}$ is the number of warmup steps
- $\gamma^0$ is `start_lr`
- $\gamma^{\text{warmup}} = f^{\text{warmup}} \cdot \gamma^0$ is the initial warmup learning rate
- $f^{\text{warmup}}$ is `warmup_start_factor`

When `warmup_start_factor` is 0.0 (default), warmup starts from 0:

```math
\gamma(\tau) = \gamma^0 \cdot \frac{\tau}{\tau^{\text{warmup}}}
```

### Specifying warmup duration

You can specify warmup duration using either `warmup_steps` (absolute) or `warmup_ratio` (relative):

- `warmup_steps`: Explicit number of warmup steps
- `warmup_ratio`: Ratio of total training steps. Computed as `int(warmup_ratio * numb_steps)`, where `numb_steps` is derived from `training.numb_steps`

These are mutually exclusive.

## Mathematical Theory

### Notation

| Symbol                 | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| $\tau$                 | Global step index (0-indexed)                        |
| $\tau^{\text{warmup}}$ | Number of warmup steps                               |
| $\tau^{\text{decay}}$  | Number of decay steps = `numb_steps - warmup_steps`  |
| $\gamma^0$             | `start_lr`: Learning rate at start of decay phase    |
| $\gamma^{\text{stop}}$ | `stop_lr`: Learning rate at end of training          |
| $f^{\text{warmup}}$    | `warmup_start_factor`: Initial warmup LR factor      |
| $s$                    | `decay_steps`: Decay period for exponential schedule |
| $r$                    | `decay_rate`: Decay rate for exponential schedule    |

### Complete warmup formula

For steps $0 \leq \tau < \tau^{\text{warmup}}$:

```math
\gamma(\tau) = f^{\text{warmup}} \cdot \gamma^0 + \frac{(1 - f^{\text{warmup}}) \cdot \gamma^0}{\tau^{\text{warmup}}} \cdot \tau
```

### Exponential decay (stepped mode)

For steps $\tau \geq \tau^{\text{warmup}}$:

```math
\gamma(\tau) = \gamma^0 \cdot r^{\left\lfloor \frac{\tau - \tau^{\text{warmup}}}{s} \right\rfloor}
```

where the decay rate $r$ is:

```math
r = \left(\frac{\gamma^{\text{stop}}}{\gamma^0}\right)^{\frac{s}{\tau^{\text{decay}}}}
```

### Exponential decay (smooth mode)

For steps $\tau \geq \tau^{\text{warmup}}$:

```math
\gamma(\tau) = \gamma^0 \cdot r^{\frac{\tau - \tau^{\text{warmup}}}{s}}
```

### Cosine annealing

For steps $\tau \geq \tau^{\text{warmup}}$:

```math
\gamma(\tau) = \gamma^{\text{stop}} + \frac{\gamma^0 - \gamma^{\text{stop}}}{2} \left(1 + \cos\left(\frac{\pi \cdot (\tau - \tau^{\text{warmup}})}{\tau^{\text{decay}}}\right)\right)
```

Equivalently, using $\alpha = \gamma^{\text{stop}} / \gamma^0$:

```math
\gamma(\tau) = \gamma^0 \cdot \left[\alpha + \frac{1 - \alpha}{2}\left(1 + \cos\left(\frac{\pi \cdot (\tau - \tau^{\text{warmup}})}{\tau^{\text{decay}}}\right)\right)\right]
```

## Migration from versions before 3.1.3

In version 3.1.2 and earlier, `start_lr` and `stop_lr`/`stop_lr_ratio` had default values and could be omitted. Starting from version 3.1.3, these parameters are **required** and must be explicitly specified.

**Configuration in version 3.1.2:**

```json
"learning_rate": {
    "type": "exp",
    "decay_steps": 5000
}
```

**Updated configuration (version 3.1.3+):**

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

## References

This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).
