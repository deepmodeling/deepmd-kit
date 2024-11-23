# DeePMD-kit devcontainer environment

This [devcontainer](https://vscode.js.cn/docs/devcontainers/devcontainer-cli) environment setups Python and C++ environment to develop DeePMD-kit.
One can setup locally or use [GitHub Codespaces](https://docs.github.com/en/codespaces) by clicking the Code button on the DeePMD-kit repository page.
The whole setup process requires about 10 minutes, so one needs to be patient.

## Python environment

The following packages are installed into the Python environment `.venv`:

- DeePMD-kit (in edit mode)
- Backends including TensorFlow, PyTorch, JAX
- LAMMPS
- MPICH
- CMake
- pre-commit (including hooks)
- Test packages including pytest
- Doc packages including sphinx

## C++ interface

The C++ interface with TensorFlow and PyTorch support is installed into `dp` directory.

When calling and debuging LAMMPS with DeePMD-kit, use the following scripts instead of the regular `lmp`:

- `.devcontainer/lmp`
- `.devcontainer/gdb_lmp`

Use the following scripts for `pytest` with LAMMPS:

- `.devcontainer/pytest_lmp`
- `.devcontainer/gdb_pytest_lmp`

## Rebuild

Usually the Python package does not need to reinstall.
But when one wants to recompile the C++ code, the following scripts can be executed.

- `.devcontainer/build_cxx.sh`
- `.devcontainer/build_py.sh`
