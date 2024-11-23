# Easy install the latest development version

DeePMD-kit is actively developed in the `devel` branch. The documentation of the [`latest`](https://docs.deepmodeling.com/projects/deepmd/en/latest/) version matches the `devel` branch.

The following is the way to install the pre-compiled packages without [building from source](./install-from-source.md). All of them are built with [GitHub Actions](../development/cicd.md).

## Install with docker

The [`devel` tag](https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit/131827568?tag=devel) is used to mark the latest development version with CUDA 12.2 support:

```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:devel
```

For CUDA 11.8 support, use the `devel_cu11` tag.

## Install with pip

Follow [the documentation for the stable version](easy-install.md#install-python-interface-with-pip), but add `--pre` and `--extra-index-url` options like below:

```sh
pip install -U --pre deepmd-kit[gpu,cu12,lmp,torch] --extra-index-url https://deepmodeling.github.io/deepmd-kit/simple
```

## Download pre-compiled C Library {{ tensorflow_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

The [pre-comiled C library](./install-from-c-library.md) can be downloaded from [here](https://nightly.link/deepmodeling/deepmd-kit/workflows/package_c/devel/libdeepmd_c-0-libdeepmd_c.tar.gz.zip), or via a shell command:

```sh
wget https://nightly.link/deepmodeling/deepmd-kit/workflows/package_c/devel/libdeepmd_c-0-libdeepmd_c.tar.gz.zip && unzip libdeepmd_c-0-libdeepmd_c.tar.gz.zip
```

## Pre-release conda-forge packages

Pre-release conda-forge packages are in `conda-forge/label/deepmd-kit_dev` or `conda-forge/label/deepmd-kit_rc` channels, other than the `conda-forge` channel.
See [conda-forge documentation](https://conda-forge.org/docs/maintainer/knowledge_base/#pre-release-builds) for more information.
