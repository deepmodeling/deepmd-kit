# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeePMD-kit is a deep learning-based molecular dynamics potential model modeling software package that supports four deep learning backends: TensorFlow, PyTorch, JAX, and Paddle, and integrates with multiple MD software including LAMMPS, i-PI, AMBER, CP2K, GROMACS, etc.

## Common Development Commands

Use this python if needed: /home/outisli/miniforge3/envs/dpmd/bin/python

### Code Check and Format

```bash
ruff check .      # Check code style
ruff format .     # Format code
isort .           # Sort imports
```

### Test Commands

```bash
# Verify installation
dp --version
python -c "import deepmd; import deepmd.tf; print('Interfaces working')"

# VITAL!: set these three OMP_NUM_THREADS, DP_INTER_OP_PARALLELISM_THREADS, DP_INTRA_OP_PARALLELISM_THREADS to zero before running test

# Single test (recommended for development)
pytest source/tests/tf/test_dp_test.py::TestDPTestEner::test_1frame -v

# Specific test suite
pytest source/tests/tf/test_dp_test.py -v

# Training test
cd examples/water/se_e2_a
dp train input.json --skip-neighbor-stat  # TensorFlow
dp --pt train input_torch.json --skip-neighbor-stat  # PyTorch
```

### Model Compression (Reference: doc/outisli/compress.md)

#### Compression Principle

- **Tabulation**: Pre-compute and store embedding network outputs
- **Piecewise Interpolation**: Use quintic Hermite interpolation for continuity
- **Performance**: Significantly reduces memory usage and improves inference speed

#### Supported Descriptors

- ✅ SE_A, SE_R, SE_T, SE_Atten
- ✅ DPA1, DPA2
- ❌ DPA3 (compression not supported)

## Code Architecture and Core Modules

### 1. Deep Learning Model Layer (deepmd/dpmodel/)

This is the core model definition layer of DeePMD-kit, containing all mathematical abstractions of models:

- **descriptor/**: Descriptor modules (embedding networks, environment information extraction)
  - `se_a.py`: Embedded Atom Descriptor
  - `se_r.py`: Simplified embedding descriptor
  - `se_a_tpe.py`: Descriptor with type embedding
  - `hybrid.py`: Hybrid descriptor
- **fitting/**: Fitting network modules
  - `ener.py`: Energy fitting network
  - `dipole.py`: Dipole fitting
  - `polar.py`: Polarizability fitting
- **model/**: Model definitions
  - `model.py`: Base model class
  - `ener_model.py`: Energy model
  - `dos_model.py`: Density of states model

### 2. Backend Implementation Layer

Each backend implements the same interface to ensure consistency:

#### TensorFlow Backend (deepmd/tf/)

- **entrypoints/**: Command line entry points
  - `main.py`: Main CLI entry
  - `train.py`: Training script
  - `freeze.py`: Model freezing
  - `test.py`: Model testing
- **network/**: Network definitions
  - `network.py`: Main network class
  - `embedding_net.py`: Embedding network
  - `fitting_net.py`: Fitting network
- **model/**: Model implementations
  - `model.py`: Model definition
  - `model_stat.py`: Model statistics
- **infer/**: Inference interface
  - `deep_eval.py`: Deep evaluation
  - `deep_pot.py`: Deep potential

#### PyTorch Backend (deepmd/pt/)

Similar structure to TensorFlow backend but with PyTorch-specific optimizations:

- **model/**: PyTorch model implementations
  - `model.py`: Base model class
  - `nn.py`: Neural network modules
- **utils/**: PyTorch utilities
  - `env_mat.py`: Environment matrix construction
  - `region.py`: Periodic boundary condition handling
- **train/**: Training related
  - `training.py`: Training loop
  - `optimizer.py`: Optimizer configuration

### 3. C++ Core Engine (source/)

Core implementation for high-performance computing:

#### Core Library (source/lib/)

- **include/**: Header file definitions
  - `deepmd.hpp`: Main API declarations
  - `common.hpp`: Common definitions
  - `neighbor_list.hpp`: Neighbor list algorithm
- **src/**: Source code implementation
  - `deepmd.cpp`: Core C++ implementation
  - `region.cpp`: Region processing
  - `neighbor_list.cpp`: High-performance neighbor list
  - `prod_env_mat_a.cpp`: Environment matrix production

#### Operator Implementation (source/op/)

Framework-specific operators for each deep learning framework:

- **tf/**: TensorFlow custom operators
  - `prod_env_mat_a.cc`: Environment matrix operator
  - `prod_force_se_a.cc`: Force calculation operator
  - `tabulate.cc`: Lookup table operator
- **torch/**: PyTorch C++ extensions
  - `prod_env_mat_a.cpp`: PyTorch version of environment matrix operator

### 4. Data Processing Layer (deepmd/utils/)

- **data.py**: Data loading and preprocessing
- `data_system.py`: Data system management
- `shuffle.py`: Data shuffling
- `neighbor_stat.py`: Neighbor statistics
- `type_embed.py`: Type embedding
- `args.py`: Argument parsing
- `path.py`: Path handling
- `compat.py`: Version compatibility handling

### 5. Input/Output Layer (deepmd/infer/)

- **deep_pot.py**: High-level inference interface
- **deep_dipole.py**: Dipole inference
- **deep_dos.py**: Density of states inference
- **deep_wfc.py**: Wave function inference

## Key Data Flow

1. **Training Flow**:

   ```
   Atomic coordinates → neighbor_list → env_matrix → descriptor → fitting_net → loss
   ```

2. **Inference Flow**:

   ```
   Input structure → Descriptor calculation → Fitting network → Energy/Force/Stress
   ```

3. **Multi-backend Unified Interface**:
   - Python layer provides unified API through `deepmd.infer`
   - C++ layer provides unified interface through `source/api_cc/`
   - Each backend implements the same model specification

### Select Backend

```bash
# Command line flags
dp --pt train input.json
dp --tf train input.json

# Environment variable
export DP_BACKEND=pytorch
dp train input.json
```

## Core Algorithms and Data Structures

### 1. Descriptor Implementation

Descriptors are the core innovation of DeePMD-kit, used to convert local atomic environments into vector representations:

#### Embedded Atom Descriptor (SE_A)

- **Location**: `deepmd/dpmodel/descriptor/se_a.py`
- **Core functions**:
  - `build()`: Build descriptor network
  - `call()`: Calculate descriptor values
- **Mathematical principle**:
  - Radial basis function expansion: $g(r) = \sum_{i} \exp[-\gamma (r-r_s)^2]$
  - Angular basis function: Angular dependency through 1D filters

#### Environment Matrix (Env Mat)

- **C++ implementation**: `source/lib/src/prod_env_mat_a.cpp`
- **Function**: Efficiently calculate environment matrix between atom pairs
- **Optimization**: Use parallelization and SIMD instructions for acceleration

### 2. Fitting Network

Maps descriptors to physical quantities:

#### Energy Fitting

- **Location**: `deepmd/dpmodel/fitting/ener.py`
- **Output**: Atomic energy, system total energy obtained by summation
- **Force calculation**: Through automatic differentiation or analytical gradient

#### Fitting Network Structure

```python
# Typical fitting network architecture
FittingNet(
    layers=[embedding_dim, 240, 240, 240, 1],  # Network layer sizes
    activation_function="tanh",  # Activation function
    precision="float64",  # Numerical precision
)
```

### 3. Training Strategy

#### Loss Function

```python
# Location: deepmd/loss.py or backend implementations
Loss = lr_e * energy_loss + lr_f * force_loss + lr_v * virial_loss
```

#### Data Preprocessing

- **Data shuffling**: `deepmd/utils/shuffle.py`
- **Batching**: Auto-fill to ensure consistent batch size
- **Data augmentation**: Increase data diversity through rotation and translation

### 4. Model Saving and Loading

#### Checkpoint Formats

- **TensorFlow**: .pb format (frozen graph)
- **PyTorch**: .pth format
- **Universal format**: .dp format (framework-agnostic)

#### Model Conversion

```python
# TensorFlow to PyTorch conversion
from deepmd.pt import model as pt_model

pt_model.load_tf_graph(tf_checkpoint_path)
```

## Common Development Patterns

### 1. Adding New Descriptors

1. Create new descriptor class in `deepmd/dpmodel/descriptor/`
2. Inherit from `BaseDescriptor` and implement necessary methods
3. Add corresponding implementations in each backend (tf/pt/jax/pd)
4. Add unit tests

### 2. Debugging Tips

- Use small systems for quick testing
- Check energy conservation and symmetry
- Compare results consistency across different backends
- Use `dp test --rand-init` to verify model

## Development Standards

### Naming Conventions

- Always use correct capitalization: DeePMD-kit, PyTorch, TensorFlow, NumPy, GitHub, LAMMPS

### License Requirements

All source files must include header license:
`SPDX-License-Identifier: LGPL-3.0-or-later`

## Test Strategy

### Test Locations

- **source/tests/**: C++ and Python tests
- **tests/** directories in each submodule

### Test Principles

- During development, only run single or few related tests; full test suite takes 60+ minutes
- Training tests use `--skip-neighbor-stat` to skip statistics for speed
- Use `timeout` to limit training test time

## Configuration File Structure

### Typical Training Configuration (input.json)

```json
{
  "model": {
    "type_map": ["O", "H"],
    "descriptor": {
      "type": "se_a",
      "sel": [46, 92],
      "rcut_smth": 5.8,
      "rcut": 6.0,
      "neuron": [25, 50, 100],
      "axis_neuron": 12
    },
    "fitting_net": {
      "type": "ener",
      "neuron": [240, 240, 240],
      "resnet_dt": true
    }
  },
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "decay_steps": 5000
  },
  "loss": {
    "start_pref_e": 0.02,
    "start_pref_f": 1000,
    "start_pref_v": 0.0
  },
  "training": {
    "training_data": {
      "systems": ["system1/", "system2/"],
      "batch_size": 8
    },
    "numb_steps": 1000000
  }
}
```

## Special Features

### 1. Type Embedding

- Support unified training for multi-element systems
- Location: `deepmd/utils/type_embed.py`
- Dynamic type embedding can handle unseen element combinations

### 2. Adaptive Selection (UpdateSel)

- Automatically update neighbor list selection parameters
- Avoid neighbor loss due to atomic migration
- Location: `deepmd/utils/update_sel.py`

### 3. Multi-task Learning

- Simultaneously fit energy, force, stress, dipole, etc.
- Loss function can configure weights for each task
- Support physical constraints and regularization

## Model Compression Details (Advanced)

### Compression Data Structure

#### 1. Compression Information (compress_info)

```python
# Store 6 parameters for each embedding network [6]
compress_info[embedding_idx] = torch.tensor(
    [
        lower[net],  # Lower bound
        upper[net],  # Upper bound
        upper[net] * extrapolate,  # Extrapolation upper bound
        table_stride_1,  # First segment stride
        table_stride_2,  # Second segment stride
        check_frequency,  # Overflow check frequency
    ]
)
```

#### 2. Compression Data (compress_data)

```python
# Store coefficient table for each embedding network [nspline, 6 * last_layer_size]
compress_data[embedding_idx] = table_data[net]

# Each 6 consecutive coefficients represent polynomial coefficients
# [f(x), f'(x), f''(x)/2, c3, c4, c5] × last_layer_size
```

### Tabulation Implementation

- **Table Builder**: `deepmd/pt/utils/tabulate.py` (PyTorch)
- **Common Utilities**: `deepmd/utils/tabulate.py`
- **Supported Activations**: tanh, gelu, relu, relu6, softplus, sigmoid

### Polynomial Interpolation Formula

In interval [x_i, x_{i+1}], for variable x, the polynomial is:

```
f(x) = c₀ + c₁t + c₂t² + c₃t³ + c₄t⁴ + c₅t⁵
```

Where:

- `t = (x - x_i) / h`, h is step size
- `c₀ = f(x_i)`
- `c₁ = f'(x_i) × h`
- `c₂ = f''(x_i) × h² / 2`
- `c₃, c₄, c₅` determined by boundary continuity
