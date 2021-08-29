# Building conda packages

One may want to keep both convenience and personalization of the DeePMD-kit. To achieve this goal, one can consider builing conda packages. We provide building scripts in [deepmd-kit-recipes organization](https://github.com/deepmd-kit-recipes/). These building tools are driven by [conda-build](https://github.com/conda/conda-build) and [conda-smithy](https://github.com/conda-forge/conda-smithy).

For example, if one wants to turn on `MPIIO` package in LAMMPS, go to [`lammps-dp-feedstock`](https://github.com/deepmd-kit-recipes/lammps-dp-feedstock/) repository and modify `recipe/build.sh`. `-D PKG_MPIIO=OFF` should be changed to `-D PKG_MPIIO=ON`. Then go to the main directory and executing

```sh
./build-locally.py
```

This requires the Docker has been installed. After the building, the packages will be generated in `build_artifacts/linux-64` and `build_artifacts/noarch`, and then one can install then execuating
```sh
conda create -n deepmd lammps-dp -c file:///path/to/build_artifacts -c https://conda.deepmodeling.org -c nvidia
```

One may also upload packages to one's Anaconda channel, so they can be installed on other machines:

```sh
anaconda upload /path/to/build_artifacts/linux-64/*.tar.bz2 /path/to/build_artifacts/noarch/*.tar.bz2
```
