# Easy install the latest development version

DeePMD-kit is actively developed in the `devel` branch. The documentation of the [`latest`](https://docs.deepmodeling.com/projects/deepmd/en/latest/) version matches the `devel` branch.

The following is the way to install the pre-compiled packages without [building from source](./install-from-source.md). All of them are built with [GitHub Actions](../development/cicd.md).

## Install with docker

The [`devel` tag](https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit/131827568?tag=devel) is used to mark the latest development version with CUDA support:

```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:devel
```

## Install with pip

Below is an one-line shell command to download the [artifact](https://nightly.link/deepmodeling/deepmd-kit/workflows/build_wheel/devel/artifact.zip) containing wheels and install it with `pip`:

```sh
pip install -U --pre deepmd-kit[gpu,cu12,lmp] --extra-index-url https://deepmodeling.github.io/deepmd-kit/simple
```

`cu12` and `lmp` are optional, which is the same as the stable version.

## Download pre-compiled C Library

The [pre-comiled C library](./install-from-c-library.md) can be downloaded from [here](https://nightly.link/deepmodeling/deepmd-kit/workflows/package_c/devel/libdeepmd_c.zip), or via a shell command:

```sh
wget https://nightly.link/deepmodeling/deepmd-kit/workflows/package_c/devel/libdeepmd_c.zip && unzip libdeepmd_c.zip
```
