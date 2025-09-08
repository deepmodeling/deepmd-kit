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
- **Main Descriptor**: `DescrptDPA3` in `deepmd/pt/model/descriptor/dpa3.py`
- **RepFlow Block**: `DescrptBlockRepflows` in `deepmd/pt/model/descriptor/repflows.py`
- **RepFlow Layer**: `RepFlowLayer` in `deepmd/pt/model/descriptor/repflow_layer.py`

#### DPA3 Initialization and Forward Pass
**Initialization** (`dpa3.py:105-171`):
- Processes RepFlow parameters
- Creates type embedding network (`TypeEmbedNetConsistent`)
- Initializes RepFlow blocks with edge/angle embedding networks
- Sets up multiple RepFlow layers for iterative refinement

**Forward Pass** (`dpa3.py:430-498`):
1. **Type Embedding**: Computes atomic type embeddings
2. **RepFlow Processing**: Multi-layer node/edge/angle information processing
3. **Output**: Returns node descriptors, rotation matrices, edge embeddings, and switch functions

**DPA3 Output Variables**:
- `node_ebd`: Node descriptors [nf, nloc, n_dim] - main atomic environment representation
- `rot_mat`: Rotation matrices [nf, nloc, e_dim, 3] - for SE(3) equivariance
- `edge_ebd`: Edge embeddings [nf, nloc, nnei, e_dim] - pairwise interactions
- `h2`: Angle information [nf, nloc, nnei, 3] - 3-body angular data
- `sw`: Switch functions [nf, nloc, nnei] - smooth cutoff boundaries

#### RepFlow Implementation
**RepFlow Block** (`repflows.py:77-200`):
- Edge embedding network for distance information
- Angle embedding network for angular information
- Multiple RepFlow layers for iterative updates
- Support for message compression and multi-head attention

**Key Parameters**:
- `e_rcut`/`e_rcut_smth`: Edge cutoff and smoothing radii
- `a_rcut`/`a_rcut_smth`: Angle cutoff and smoothing radii  
- `n_dim`/`e_dim`/`a_dim`: Node/edge/angle representation dimensions
- `nlayers`: Number of RepFlow layers
- `update_style`: Residual connection strategies (res_residual, res_update, etc.)

#### CLI Usage and Training Flow
**Training Command**: `dp train input.json`

**Execution Flow**:
1. **Entry Point**: `deepmd.pt.entrypoints.main.train()` (`main.py:237-248`)
2. **Configuration Loading**: JSON parsing and multi-task handling
3. **Neighbor Statistics**: Automatic selection parameter computation
4. **Trainer Creation**: `get_trainer()` with model initialization
5. **Model Building**: DPA3 descriptor creation via `get_model()`

#### Precision Control
DPA3 supports two levels of precision control:

**Environment Variable Control**:
```bash
export DP_INTERFACE_PREC=high  # Default: float64 interface
export DP_INTERFACE_PREC=low   # Lower memory: float32 interface
```

**Model Parameter Control**:
```json
{
  "model": {
    "descriptor": {
      "type": "dpa3",
      "precision": "float32",
      "repflow": {
        "precision": "float32"
      }
    }
  }
}
```

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
```python
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

### Common Pitfalls
- Backend-specific imports are banned at module level (use runtime imports)
- Model compatibility requires careful version management
- GPU builds require specific CUDA/ROCm versions

## File Structure Conventions

- **Public APIs**: In `deepmd/` top-level modules
- **Implementation details**: In subdirectories like `dpmodel/`, `utils/`
- **Backend code**: Separated into `tf/`, `pt/`, `jax/`, `pd/` directories
- **Tests**: Organized by backend in `source/tests/*/`
- **Examples**: In `examples/` directory with input configurations