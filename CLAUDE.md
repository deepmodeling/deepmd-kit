# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeePMD-kit is a deep learning package for many-body potential energy representation and molecular dynamics. It supports multiple backends (TensorFlow, PyTorch, JAX, Paddle) and interfaces with various MD packages (LAMMPS, i-PI, AMBER, GROMACS, etc.).

## Development Commands

### Building and Installation

- **Standard build**: `pip install .`
- **With GPU support**: Set environment variables like `DP_ENABLE_PYTORCH=1`, `DP_ENABLE_TENSORFLOW=1`, etc.
- **From source**: Uses scikit-build-core with CMake - see `source/CMakeLists.txt`
- **C++ library**: Built automatically as part of the Python package

### Testing

- **Run all tests**: `pytest source/tests`
- **Run specific backend tests**: `pytest source/tests/tf/`, `pytest source/tests/pt/`, etc.
- **GPU tests**: `tox -e gpu` or set `DP_VARIANT=cuda`
- **Individual test**: `pytest source/tests/path/to/test_file.py::test_name`
- **With coverage**: `pytest --cov=deepmd`

### Code Quality

- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type checking**: No specific type checker configured in the project

### Backend-Specific Commands

- **TensorFlow**: Requires TF 2.19.0, automatically enabled with certain flags
- **PyTorch**: Enable with `DP_ENABLE_PYTORCH=1`
- **JAX**: Enable with `DP_ENABLE_JAX=1` (requires Python >= 3.10)
- **Paddle**: Enable with `DP_ENABLE_PADDLE=1`

### Model Compression

- **Compress models**: `dp --pt compress -i model.pth -o compressed.pth`
- **Custom parameters**: `dp --pt compress -i model.pth -o compressed.pth -s 0.005 -e 10`
- **PyTorch backend only**: Supports SE_A, SE_R, SE_T, SE_Atten, DPA1, DPA2 descriptors
- **DPA3 not supported**: Compression explicitly disabled for DPA3 descriptors

## Architecture Overview

### Multi-Backend Design

The codebase is organized around a modular backend system in `deepmd/backend/`:

- `backend.py`: Core backend management logic
- `tensorflow.py`, `pytorch.py`, `jax.py`, `paddle.py`: Backend-specific implementations
- `suffix.py`: Model file suffix handling for different backends

### Core Components

#### 1. Model Architecture (`deepmd/dpmodel/`)

Framework-agnostic model implementations:

- `atomic_model/`: Atomic-level model components
- `descriptor/`: Environment descriptors (se_a, se_atten, dpa1/2/3, etc.)
- `fitting/`: Fitting networks for energy, forces, etc.
- `model/`: Complete model definitions

### DPA3 Descriptor Implementation

#### DPA3 Architecture Overview

DPA3 (Deep Potential - Atomic Environment Representation with 3-body interactions) is an advanced descriptor that combines node, edge, and angle information for more accurate atomic environment representation.

**Key Components**:

- **Main Descriptor**: `DescrptDPA3` in `deepmd/pt/model/descriptor/dpa3.py:105-171`
- **RepFlow Block**: `DescrptBlockRepflows` in `deepmd/pt/model/descriptor/repflows.py:77-200`
- **RepFlow Layer**: `RepFlowLayer` in `deepmd/pt/model/descriptor/repflow_layer.py:38-200`

**DPA3 Core Innovation**: The RepFlow architecture introduces a unified representation that iteratively refines node, edge, and angle information through multiple layers, enabling explicit 3-body interaction modeling while maintaining computational efficiency through message compression strategies.

#### DPA3 Initialization and Forward Pass

**Initialization** (`dpa3.py:105-171`):

- Processes RepFlow parameters with `init_subclass_params(repflow, RepFlowArgs)`
- Creates type embedding network (`TypeEmbedNetConsistent`) for consistent atomic type representations
- Initializes RepFlow blocks with edge/angle embedding networks for distance and angular information
- Sets up multiple RepFlow layers for iterative refinement with configurable residual connections

**Forward Pass** (`dpa3.py:430-498`):

1. **Type Embedding**: Computes atomic type embeddings using `TypeEmbedNetConsistent`
2. **RepFlow Processing**: Multi-layer node/edge/angle information processing through iterative updates
3. **Output Generation**: Returns comprehensive atomic environment representation with rotation matrices for SE(3) equivariance

**DPA3 Output Variables**:

- `node_ebd`: Node descriptors [nf, nloc, n_dim] - primary atomic environment representation for fitting networks
- `rot_mat`: Rotation matrices [nf, nloc, e_dim, 3] - ensures SE(3) equivariance for coordinate transformations
- `edge_ebd`: Edge embeddings [nf, nloc, nnei, e_dim] - pairwise interaction information
- `h2`: Angle information [nf, nloc, nnei, 3] - 3-body angular data for explicit three-body interactions
- `sw`: Switch functions [nf, nloc, nnei] - smooth cutoff boundaries to avoid discontinuities

#### RepFlow Implementation

**RepFlow Block** (`repflows.py:77-200`):

- Edge embedding network (`MLPLayer`) for distance information encoding
- Angle embedding network for angular relationship processing
- Multiple RepFlow layers (`RepFlowLayer`) for iterative node/edge/angle updates
- Support for message compression (`a_compress_rate`) and attention mechanisms to reduce computational cost
- Environment matrix computation via `prod_env_mat` for neighbor distance and direction calculation

**Key Parameters**:

- `e_rcut`/`e_rcut_smth`: Edge cutoff (6.0Å) and smoothing radii (0.5Å) for neighbor selection
- `a_rcut`/`a_rcut_smth`: Angle cutoff (4.0Å) and smoothing radii for three-body interactions
- `n_dim`/`e_dim`/`a_dim`: Node (128), edge (64), angle (32) representation dimensions
- `nlayers`: Number of RepFlow layers (6) for iterative refinement
- `update_style`: Residual connection strategies (`res_residual`, `res_update`, `force_residual`) for gradient flow optimization
- `a_compress_rate`: Angle compression factor (2) to reduce computational overhead while preserving angular information

#### CLI Usage and Training Flow

**Training Command**: `dp --pt train input.json` (specify PyTorch backend explicitly)

**Execution Flow**:

1. **Entry Point**: `deepmd.pt.entrypoints.main.train()` (`main.py:248-372`) - PyTorch-specific training entry
2. **Configuration Loading**: JSON parsing via `j_loader()` with multi-task handling through `preprocess_shared_params()`
3. **Neighbor Statistics**: Automatic selection parameter computation via `BaseModel.update_sel()` unless `--skip-neighbor-stat`
4. **Trainer Creation**: `get_trainer()` with model initialization, supporting distributed training and mixed precision
5. **Model Building**: DPA3 descriptor creation via `get_model()` with automatic device placement and JIT compilation options

**Data Processing Pipeline**:

1. **Raw Data Loading**: `DeepmdData` loads HDF5/.npy files from system directories
2. **System DataLoaders**: Each system gets its own DataLoader (num_workers=0 to avoid thread explosion)
3. **Training DataLoader**: Master DataLoader with intelligent sampling (`WeightedRandomSampler` or uniform)
4. **Batch Processing**: `collate_batch()` handles variable-sized systems with padding and tensor stacking

#### Precision Control

DPA3 supports two levels of precision control that work independently:

**Environment Variable Control (`DP_INTERFACE_PREC`)**:

- **Scope**: Global interface precision affecting input/output data types across all DeePMD-kit operations
- **High precision** (`export DP_INTERFACE_PREC=high`): `GLOBAL_NP_FLOAT_PRECISION = np.float64`, `GLOBAL_ENER_FLOAT_PRECISION = np.float64`
- **Low precision** (`export DP_INTERFACE_PREC=low`): `GLOBAL_NP_FLOAT_PRECISION = np.float32`, `GLOBAL_ENER_FLOAT_PRECISION = np.float64` (energy precision remains high)
- **Location**: `deepmd/env.py:33-48`

**Model Parameter Control (`precision` in configuration)**:

- **Scope**: Component-specific precision for neural network weights and calculations
- **Options**: `"float64"`, `"float32"`, `"float16"`, `"default"`
- **Granular Control**: Can be set individually for descriptor, fitting networks, and RepFlow components
- **Example Configuration**:

```json
{
  "model": {
    "descriptor": {
      "type": "dpa3",
      "precision": "float32",
      "repflow": {
        "precision": "float32"
      }
    },
    "fitting_net": {
      "precision": "float32"
    }
  }
}
```

**Precision Workflow** (`make_model.py:327-337`):

1. **Input Type Detection**: `input_type_cast()` detects input data precision
2. **Global Precision Conversion**: Converts to `GLOBAL_PT_FLOAT_PRECISION` for computation
3. **Component Computation**: Uses component-specific precision settings
4. **Output Conversion**: `output_type_cast()` converts back to original input precision

#### Inference System

**Main Classes**:

- `DeepEval`: Universal inference interface (`deepmd/pt/infer/deep_eval.py:75`)
- `Tester`: Testing and inference utility (`deepmd/pt/infer/inference.py:25`)

**Inference Flow**:

1. **Model Loading**: State dict loading and multi-task handling
2. **JIT Compilation**: Optional TorchScript optimization
3. **Batch Processing**: Automatic batch sizing for memory optimization
4. **Execution**: DPA3 descriptor computation in evaluation mode

**Performance Optimizations**:

- **JIT Compilation**: `torch.jit.script()` for graph optimization
- **Auto-batching**: Dynamic batch size adjustment based on memory
- **Multi-device**: CPU/GPU support with automatic device selection
- **Model Freezing**: `dp freeze` for deployment-optimized models

#### Configuration Example

```json
{
  "model": {
    "descriptor": {
      "type": "dpa3",
      "repflow": {
        "e_rcut": 6.0,
        "e_sel": 120,
        "a_rcut": 4.0,
        "a_sel": 40,
        "n_dim": 128,
        "e_dim": 64,
        "a_dim": 32,
        "nlayers": 3,
        "update_style": "res_residual"
      },
      "concat_output_tebd": true,
      "precision": "float32"
    }
  }
}
```

#### Energy Summation Mechanism

DPA3 implements a two-stage energy calculation:

1. **Atomic Energy**: Each atom's local environment energy computed in fitting networks
2. **System Energy**: Atomic energies summed to get total system energy

**Key Files**:

- Atomic energy: `deepmd/pt/model/task/fitting.py:473-614`
- Energy summation: `deepmd/pt/model/model/transform_output.py:153-192`

### Model Compression System

#### Compression Overview

DeePMD-kit supports model compression through tabulation of embedding networks, providing significant inference speedup by replacing neural network computations with polynomial interpolation lookups.

**Core Concept**:

- Pre-compute embedding network outputs and store in lookup tables
- Use two-stage interpolation with different stride sizes for accuracy-memory balance
- Replace runtime neural network evaluations with fast polynomial interpolation

#### Compression Architecture

**Entry Points**:

- Command: `dp --pt compress -i model.pth -o compressed.pth`
- Main entry: `deepmd/main.py` → `deepmd/pt/entrypoints/main.py:574-582`
- Core function: `deepmd/pt/entrypoints/compress.py:32-84`

**Execution Flow**:

1. **Model Loading**: Load JIT model and reconstruct model instance
2. **Min Distance Calculation**: Compute minimum neighbor distance from training data
3. **Hierarchical Compression**: Model → Atomic Model → Descriptor compression
4. **Table Building**: Create polynomial coefficient tables via `DPTabulate`
5. **JIT Serialization**: Save compressed model as TorchScript

#### Supported Descriptors

**Fully Supported**:

- `SE_A` (`se_a.py:257-302`): Smooth Edition Angular descriptor
- `SE_R` (`se_r.py:359-xxx`): Smooth Edition Radial descriptor
- `SE_T` (`se_t.py:284-327`): Smooth Edition Three-body descriptor
- `SE_Atten` (`se_atten.py:427-448`): Smooth Edition with Attention
- `DPA1` (`dpa1.py:572-645`): Deep Potential Attention version 1
- `DPA2` (`dpa2.py:893-973`): Deep Potential Attention version 2

**Not Supported**:

- `DPA3` (`dpa3.py:578-601`): Explicitly raises `NotImplementedError`
- `Pairtab` models: No tabulation compression support

#### Tabulation Implementation

**Key Class**: `DPTabulate` (`deepmd/pt/utils/tabulate.py:30-100`)

**Table Building Process**:

1. **Range Calculation**: Compute environment matrix bounds from training data statistics
2. **Grid Generation**: Create two-segment distance grids (fine + coarse stride)
3. **Neural Network Evaluation**: Forward pass to get function values and derivatives
4. **Polynomial Fitting**: Generate 5th-order Hermite interpolation coefficients

**Data Storage Format**:

- `compress_info`: [lower, upper, extrapolate_upper, stride1, stride2, check_freq]
- `compress_data`: [nspline, 6 * last_layer_size] coefficient tables
- Coefficients: [f(x), f'(x), f''(x)/2, c3, c4, c5] per neuron

#### Performance Characteristics

**Memory Optimization**:

- Two-stage interpolation: fine stride (0.01) + coarse stride (0.1)
- Extrapolation region: 5× training data range by default
- Removes original network weights after compression

**Computational Benefits**:

- Eliminates matrix operations in embedding networks
- Vectorized polynomial evaluation
- Cache-friendly data layout for lookup tables

#### Configuration Parameters

- `-s, --step`: Fine stride size (default: 0.01) - affects accuracy vs memory
- `-e, --extrapolate`: Extrapolation multiplier (default: 5)
- `-f, --frequency`: Overflow check frequency (default: -1, disabled)
- `-t, --training-script`: Training script path for min distance calculation

#### 2. Backend-Specific Implementations

- `deepmd/tf/`: TensorFlow backend (original implementation)
- `deepmd/pt/`: PyTorch backend
- `deepmd/jax/`: JAX backend
- `deepmd/pd/`: Paddle backend

Each backend implements similar interfaces:

- Descriptor variants optimized for the framework
- Training and inference modules
- Model serialization/loading

#### 3. Inference (`deepmd/infer/`)

High-level inference interfaces:

- `deep_pot.py`: Main potential energy model interface
- `deep_eval.py`: Generic evaluation interface
- Backend-specific inference modules

#### 4. Training (`deepmd/*/train/`)

Backend-specific training implementations:

- Training loops and optimization
- Data loading and preprocessing
- Checkpoint management

#### 5. Entry Points (`deepmd/entrypoints/`)

Command-line interface commands:

- `main.py`: Main CLI dispatcher
- Training, testing, conversion utilities
- Model analysis and documentation tools

#### 6. C++ Integration (`source/`)

- `lib/`: Core computational library with CUDA/ROCm support
- `api_cc/`: C++ API for external integration
- `api_c/`: C API wrapper
- `lmp/`: LAMMPS plugin integration
- `op/`: Custom operators for different frameworks

### PyTorch Backend Data Processing

#### Two-Level DataLoader Architecture

The PyTorch backend uses a unique two-level DataLoader system for efficient multi-system data management:

**System Level**: Each data system has its own DataLoader (num_workers=0 to avoid thread explosion)
**Training Level**: Master DataLoader handles sampling and batching across systems (num_workers=NUM_WORKERS)

**Key Components**:

- `DeepmdData`: Raw data loading from HDF5/.npy files (`deepmd/utils/data.py`)
- `DpLoaderSet`: System-level DataLoader collection (`deepmd/pt/utils/dataloader.py`)
- `DeepmdDataSetForLoader`: PyTorch Dataset wrapper
- `collate_batch`: Batch processing function for variable-sized systems

**Data Flow**:

```
Raw Data (HDF5/.npy) → DeepmdData → System DataLoaders → DpLoaderSet → Training DataLoader → Model Input
```

### DPAtomicModel Hierarchy

#### Class Structure

```text
BaseAtomicModel (base_atomic_model.py:52)
    ↓
DPAtomicModel (dp_atomic_model.py:34) - registered as "standard"
    ↓
Specific Models (Energy, Dipole, Polar, DOS, Property)
```

**Key Features**:

- **Unified Interface**: Consistent API for different physical properties
- **Atomic-Level Forward Pass**: `forward_atomic()` method handles descriptor computation and fitting
- **Multi-Task Support**: Supports training multiple properties simultaneously
- **Automatic Differentiation**: Force and virial computation through autograd

**Key Files**:

- Base class: `deepmd/pt/model/atomic_model/dp_atomic_model.py:34`
- Energy model: `deepmd/pt/model/atomic_model/energy_atomic_model.py:13`
- Dipole model: `deepmd/pt/model/atomic_model/dipole_atomic_model.py:14`

### Key Design Patterns

#### Backend Abstraction

The code uses a sophisticated backend system that allows:

- Runtime backend selection
- Model conversion between backends
- Consistent APIs across frameworks

#### Descriptor-Based Architecture

Models are built from:

1. **Descriptors**: Local atomic environment representations
2. **Fitting Networks**: Map descriptors to physical quantities
3. **Models**: Combine descriptors and fitting for complete potentials

#### Multi-Task Learning

Support for training multiple properties simultaneously:

- Energy, forces, virial
- Dipole moments, polarizability
- DOS, electronic properties
- Spin systems

## Working with the Code

### Adding New Features

1. **Framework-agnostic**: Add to `deepmd/dpmodel/` first
2. **Backend implementations**: Extend each backend in `deepmd/*/`
3. **C++ optimization**: Add performance-critical code to `source/lib/`
4. **Tests**: Add backend-specific tests in `source/tests/*/`

### Model Development

- Use existing descriptors as templates in `deepmd/dpmodel/descriptor/`
- Extend fitting networks in `deepmd/dpmodel/fitting/`
- Model composition follows patterns in `deepmd/dpmodel/model/`

### Performance Considerations

- C++ library handles neighbor lists and environment matrices
- Custom operators optimized for GPU acceleration
- Automatic mixed precision support where available
- **Model compression**: Tabulation provides 2-10× inference speedup for supported descriptors

### Common Pitfalls

- Backend-specific imports are banned at module level (use runtime imports)
- Model compatibility requires careful version management
- GPU builds require specific CUDA/ROCm versions
- **Compression limitations**: DPA3 and some specialized models don't support compression
- **Training data dependency**: Compression requires training script for optimal table range calculation

## File Structure Conventions

- **Public APIs**: In `deepmd/` top-level modules
- **Implementation details**: In subdirectories like `dpmodel/`, `utils/`
- **Backend code**: Separated into `tf/`, `pt/`, `jax/`, `pd/` directories
- **Tests**: Organized by backend in `source/tests/*/`
- **Examples**: In `examples/` directory with input configurations
