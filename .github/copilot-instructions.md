# DeePMD-kit

DeePMD-kit is a deep learning package for many-body potential energy representation and molecular dynamics. It supports multiple backends (TensorFlow, PyTorch, JAX, Paddle) and integrates with MD packages like LAMMPS, GROMACS, and i-PI.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Build Repository

- Create virtual environment: `uv venv venv && source venv/bin/activate`
- Install base dependencies: `uv pip install tensorflow-cpu` (takes ~8 seconds)
- Install PyTorch: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` (takes ~5 seconds)
- Build Python package: `uv pip install -e .[cpu,test]` -- takes 67 seconds. **NEVER CANCEL. Set timeout to 120+ seconds.**
- Build C++ components: `export TENSORFLOW_ROOT=$(python -c 'import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec("tensorflow").origin).parent)')` then `export PYTORCH_ROOT=$(python -c 'import torch;print(torch.__path__[0])')` then `./source/install/build_cc.sh` -- takes 164 seconds. **NEVER CANCEL. Set timeout to 300+ seconds.**

### Test Repository

- Run single test: `pytest source/tests/tf/test_dp_test.py::TestDPTestEner::test_1frame -v` -- takes 8-13 seconds
- Run test subset: `pytest source/tests/tf/test_dp_test.py -v` -- takes 15 seconds. **NEVER CANCEL. Set timeout to 60+ seconds.**
- **Recommended: Use single test cases for validation instead of full test suite** -- full suite has 314 test files and takes 60+ minutes

### Lint and Format Code

- Install linter: `uv pip install ruff`
- Run linting: `ruff check .` -- takes <1 second
- Format code: `ruff format .` -- takes <1 second
- **Always run `ruff check .` and `ruff format .` before committing changes or the CI will fail.**

### Training and Validation

- Test TensorFlow training: `cd examples/water/se_e2_a && dp train input.json --skip-neighbor-stat` -- training proceeds but is slow on CPU
- Test PyTorch training: `cd examples/water/se_e2_a && dp --pt train input_torch.json --skip-neighbor-stat` -- training proceeds but is slow on CPU
- **Training examples are for validation only. Real training takes hours/days. Timeout training tests after 60 seconds for validation.**

## Validation Scenarios

**ALWAYS manually validate any new code through at least one complete scenario:**

### Basic Functionality Validation

1. **CLI Interface**: Run `dp --version` and `dp -h` to verify installation
2. **Python Interface**: Run `python -c "import deepmd; import deepmd.tf; print('Both interfaces work')"`
3. **Backend Selection**: Test `dp --tf -h`, `dp --pt -h`, `dp --jax -h`, `dp --pd -h`

### Training Workflow Validation

1. **TensorFlow Training**: `cd examples/water/se_e2_a && timeout 60 dp train input.json --skip-neighbor-stat` -- should start training and show decreasing loss
2. **PyTorch Training**: `cd examples/water/se_e2_a && timeout 60 dp --pt train input_torch.json --skip-neighbor-stat` -- should start training and show decreasing loss
3. **Verify training output**: Look for "batch X: trn: rmse" messages showing decreasing error values

### Test-Based Validation

1. **Core Tests**: `pytest source/tests/tf/test_dp_test.py::TestDPTestEner::test_1frame -v` -- should pass in ~10 seconds
2. **Multi-backend**: Test both TensorFlow and PyTorch components work

## Common Commands and Timing

### Repository Structure

```
ls -la [repo-root]
.github/               # GitHub workflows and templates
CONTRIBUTING.md        # Contributing guide
README.md             # Project overview
deepmd/               # Python package source
doc/                  # Documentation
examples/             # Training examples and configurations
pyproject.toml        # Python build configuration
source/               # C++ source code and tests
```

### Key Directories and Files

- `deepmd/` - Main Python package with backend implementations
- `source/lib/` - Core C++ library
- `source/op/` - Backend-specific operators (TF, PyTorch, etc.)
- `source/api_cc/` - C++ API
- `source/api_c/` - C API
- `source/tests/` - Test suite (314 test files)
- `examples/water/se_e2_a/` - Basic water training example
- `examples/` - Various model examples for different scenarios

### Common CLI Commands

- `dp --version` - Show version information
- `dp -h` - Show help and available commands
- `dp train input.json` - Train a model (TensorFlow backend)
- `dp --pt train input.json` - Train with PyTorch backend
- `dp --jax train input.json` - Train with JAX backend
- `dp --pd train input.json` - Train with Paddle backend
- `dp test -m model.pb -s system/` - Test a trained model
- `dp freeze -o model.pb` - Freeze/save a model

### Build Dependencies and Setup

- **Python 3.9+** required
- **Virtual environment** strongly recommended: `uv venv venv && source venv/bin/activate`
- **Backend dependencies**: TensorFlow, PyTorch, JAX, or Paddle (install before building)
- **Build tools**: CMake, C++ compiler, scikit-build-core
- **C++ build requires**: Both TensorFlow and PyTorch installed, set TENSORFLOW_ROOT and PYTORCH_ROOT environment variables

### Key Configuration Files

- `pyproject.toml` - Python build configuration and dependencies
- `source/CMakeLists.txt` - C++ build configuration
- `examples/water/se_e2_a/input.json` - Basic TensorFlow training config
- `examples/water/se_e2_a/input_torch.json` - Basic PyTorch training config

## Frequent Patterns and Time Expectations

### Installation and Build Times

- **Virtual environment setup**: ~5 seconds
- **TensorFlow CPU install**: ~8 seconds
- **PyTorch CPU install**: ~5 seconds
- **Python package build**: ~67 seconds. **NEVER CANCEL.**
- **C++ components build**: ~164 seconds. **NEVER CANCEL.**
- **Full fresh setup**: ~3-4 minutes total

### Testing Times

- **Single test**: 8-13 seconds
- **Test file (~5 tests)**: ~15 seconds
- **Backend-specific test subset**: 15-30 minutes. **Use sparingly.**
- **Full test suite (314 files)**: 60+ minutes. **Avoid in development - use single tests instead.**

### Linting and Formatting

- **Ruff check**: <1 second
- **Ruff format**: <1 second
- **Pre-commit hooks**: May have network issues, use individual tools

### Commit Messages and PR Titles

**All commit messages and PR titles must follow [conventional commit specification](https://www.conventionalcommits.org/):**

- **Format**: `type(scope): description`
- **Common types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`
- **Examples**:
  - `feat(core): add new descriptor type`
  - `fix(tf): resolve memory leak in training`
  - `docs: update installation guide`
  - `ci: add workflow for testing`

### Training and Model Operations

- **Training initialization**: 10-30 seconds
- **Training per batch**: 0.1-1 second (CPU), much faster on GPU
- **Model freezing**: 5-15 seconds
- **Model testing**: 10-30 seconds

## Backend-Specific Notes

### TensorFlow Backend

- **Default backend** when no flag specified
- **Configuration**: Use `input.json` format
- **Training**: `dp train input.json`
- **Requirements**: `tensorflow` or `tensorflow-cpu` package

### PyTorch Backend

- **Activation**: Use `--pt` flag or `export DP_BACKEND=pytorch`
- **Configuration**: Use `input_torch.json` format typically
- **Training**: `dp --pt train input_torch.json`
- **Requirements**: `torch` package

### JAX Backend

- **Activation**: Use `--jax` flag
- **Training**: `dp --jax train input.json`
- **Requirements**: `jax` and related packages
- **Note**: Experimental backend, may have limitations

### Paddle Backend

- **Activation**: Use `--pd` flag
- **Training**: `dp --pd train input.json`
- **Requirements**: `paddlepaddle` package
- **Note**: Less commonly used

## Critical Warnings

- **NEVER CANCEL BUILD OPERATIONS**: Python build takes 67 seconds, C++ build takes 164 seconds
- **USE SINGLE TESTS FOR VALIDATION**: Run individual tests instead of full test suite for faster feedback
- **ALWAYS activate virtual environment**: Build and runtime failures occur without proper environment
- **ALWAYS install backend dependencies first**: TensorFlow/PyTorch required before building C++ components
- **ALWAYS run linting before commits**: `ruff check . && ruff format .` or CI will fail
- **ALWAYS test both Python and C++ components**: Some features require both to be built
- **ALWAYS follow conventional commit format**: All commit messages and PR titles must use conventional commit specification (`type(scope): description`)
