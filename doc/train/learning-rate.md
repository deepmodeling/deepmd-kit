# Learning rate

DeePMD-kit supports two learning rate schedules:

- **`exp`**: Exponential decay with optional stepped or smooth mode
- **`cosine`**: Cosine annealing for smooth decay curve

Both schedules support an optional warmup phase where the learning rate gradually increases from a small initial value to the target {ref}`start_lr <learning_rate/start_lr>`.

This page focuses on schedule behavior, examples, and formulas. For the canonical argument definitions, see {ref}`learning_rate <learning_rate>`.

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

Use {ref}`learning_rate <learning_rate>` as the canonical parameter reference. This page only highlights the argument combinations that matter when choosing a schedule:

- Shared by both {ref}`exp <learning_rate[exp]>` and {ref}`cosine <learning_rate[cosine]>`: {ref}`start_lr <learning_rate/start_lr>` plus exactly one of {ref}`stop_lr <learning_rate/stop_lr>` or {ref}`stop_lr_ratio <learning_rate/stop_lr_ratio>`.
- Optional warmup for both schedules: {ref}`warmup_steps <learning_rate/warmup_steps>` or {ref}`warmup_ratio <learning_rate/warmup_ratio>`, with optional {ref}`warmup_start_factor <learning_rate/warmup_start_factor>`.
- Optional distributed scaling for both schedules: {ref}`scale_by_worker <learning_rate/scale_by_worker>`.
- Additional options for {ref}`exp <learning_rate[exp]>`: {ref}`decay_steps <learning_rate[exp]/decay_steps>`, {ref}`decay_rate <learning_rate[exp]/decay_rate>`, and {ref}`smooth <learning_rate[exp]/smooth>`.
- {ref}`cosine <learning_rate[cosine]>` has no extra schedule-specific arguments beyond the shared ones.

See [Mathematical Theory](#mathematical-theory) for complete formulas.

## Exponential Decay Schedule

The exponential decay schedule reduces the learning rate exponentially over training steps. It is the default schedule when {ref}`type <learning_rate/type>` is omitted.

### Stepped vs smooth mode

By setting {ref}`smooth <learning_rate[exp]/smooth>` to `true`, the learning rate decays smoothly at every step instead of in a stepped manner. This provides a more gradual decay curve similar to PyTorch's `ExponentialLR`, whereas the default stepped mode (`smooth: false`) is similar to PyTorch's `StepLR`.

If {ref}`decay_rate <learning_rate[exp]/decay_rate>` is not explicitly provided, DeePMD-kit computes it from {ref}`start_lr <learning_rate/start_lr>` and the requested final learning rate so that the schedule reaches the target by {ref}`numb_steps <training/numb_steps>`. The exact expression is given in [Mathematical Theory](#mathematical-theory).

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

**Using {ref}`stop_lr_ratio <learning_rate/stop_lr_ratio>`:**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3,
    "decay_steps": 5000
}
```

Equivalent to `stop_lr: 1e-6` (i.e., `0.001 * 1e-3`).

**With warmup (using {ref}`warmup_steps <learning_rate/warmup_steps>`):**

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

Learning rate starts from `0.0001` (i.e., {ref}`warmup_start_factor <learning_rate/warmup_start_factor>` `*` {ref}`start_lr <learning_rate/start_lr>`), increases linearly to {ref}`start_lr <learning_rate/start_lr>` over 10,000 steps, then decays exponentially.

**With warmup (using {ref}`warmup_ratio <learning_rate/warmup_ratio>`):**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr_ratio": 1e-3,
    "decay_steps": 5000,
    "warmup_ratio": 0.05
}
```

If {ref}`numb_steps <training/numb_steps>` is 1,000,000, warmup lasts 50,000 steps. Learning rate starts from `0.0` (default {ref}`warmup_start_factor <learning_rate/warmup_start_factor>`) and increases to {ref}`start_lr <learning_rate/start_lr>`.

**Smooth exponential decay (with {ref}`smooth <learning_rate[exp]/smooth>`):**

```json
"learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 5000,
    "smooth": true
}
```

With {ref}`smooth <learning_rate[exp]/smooth>` set to `true`, the learning rate decays continuously at every step, similar to PyTorch's `ExponentialLR`. The default stepped mode (`smooth: false`) is similar to PyTorch's `StepLR`.

## Cosine Annealing Schedule

The cosine annealing schedule smoothly decreases the learning rate following a cosine curve. It often provides better convergence than exponential decay.

After warmup, the learning rate follows a cosine curve from {ref}`start_lr <learning_rate/start_lr>` to {ref}`stop_lr <learning_rate/stop_lr>` or the value implied by {ref}`stop_lr_ratio <learning_rate/stop_lr_ratio>`. The exact expression is given in [Mathematical Theory](#mathematical-theory).

### Examples

**Basic cosine annealing:**

```json
"learning_rate": {
    "type": "cosine",
    "start_lr": 0.001,
    "stop_lr": 1e-6
}
```

**Using {ref}`stop_lr_ratio <learning_rate/stop_lr_ratio>`:**

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

Warmup is a technique to stabilize training in early stages by gradually increasing the learning rate from {ref}`warmup_start_factor <learning_rate/warmup_start_factor>` `*` {ref}`start_lr <learning_rate/start_lr>` to {ref}`start_lr <learning_rate/start_lr>`.

You can specify warmup duration using either {ref}`warmup_steps <learning_rate/warmup_steps>` (absolute) or {ref}`warmup_ratio <learning_rate/warmup_ratio>` (relative to {ref}`numb_steps <training/numb_steps>`). These are mutually exclusive.

The exact piecewise warmup formula is given in [Mathematical Theory](#mathematical-theory).

## Mathematical Theory

### Notation

| Symbol                 | Description                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| $\tau$                 | Global step index (0-indexed)                                                              |
| $\tau^{\text{warmup}}$ | Number of warmup steps                                                                     |
| $\tau^{\text{decay}}$  | Number of decay steps = `numb_steps - warmup_steps`                                        |
| $\gamma^0$             | {ref}`start_lr <learning_rate/start_lr>`: Learning rate at start of decay phase            |
| $\gamma^{\text{stop}}$ | {ref}`stop_lr <learning_rate/stop_lr>`: Learning rate at end of training                   |
| $f^{\text{warmup}}$    | {ref}`warmup_start_factor <learning_rate/warmup_start_factor>`: Initial warmup LR factor   |
| $s$                    | {ref}`decay_steps <learning_rate[exp]/decay_steps>`: Decay period for exponential schedule |
| $r$                    | {ref}`decay_rate <learning_rate[exp]/decay_rate>`: Decay rate for exponential schedule     |

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

In version 3.1.2 and earlier, {ref}`start_lr <learning_rate/start_lr>` and {ref}`stop_lr <learning_rate/stop_lr>` / {ref}`stop_lr_ratio <learning_rate/stop_lr_ratio>` had default values and could be omitted. Starting from version 3.1.3, these parameters are **required** and must be explicitly specified.

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
