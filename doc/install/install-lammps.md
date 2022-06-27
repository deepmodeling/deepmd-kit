# Install LAMMPS

There are two ways to install LAMMPS: the built-in mode and the plugin mode. The built-in mode builds LAMMPS along with the DeePMD-kit and DeePMD-kit will be loaded automatically when running LAMMPS. The plugin mode builds LAMMPS and a plugin separately, so one needs to use `plugin load` command to load the DeePMD-kit's LAMMPS plugin library. 

## Install LAMMPS's DeePMD-kit module (built-in mode)
Before following this section, [DeePMD-kit C++ interface](install-from-source.md) should have be installed.

DeePMD-kit provides a module for running MD simulation with LAMMPS. Now make the DeePMD-kit module for LAMMPS.

```bash
cd $deepmd_source_dir/source/build
make lammps
```
DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory. If you need the low precision version, move `env_low.sh` to `env.sh` in the directory. Now download the LAMMPS code, and uncompress it. The LAMMPS version should be the same as what is specified as the CMAKE argument `LAMMPS_VERSION_NUMBER`.
```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_23Jun2022.tar.gz
tar xf stable_23Jun2022.tar.gz
```
The source code of LAMMPS is stored in directory `lammps-stable_23Jun2022`. Now go into the LAMMPS code and copy the DeePMD-kit module like this
```bash
cd lammps-stable_23Jun2022/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
make yes-kspace
make yes-user-deepmd
```
You can enable any other package you want. Now build LAMMPS
```bash
make mpi -j4
```

If everything works fine, you will end up with an executable `lmp_mpi`.
```bash
./lmp_mpi -h
```

The DeePMD-kit module can be removed from LAMMPS source code by 
```bash
make no-user-deepmd
```

## Install LAMMPS (plugin mode)
Starting from `8Apr2021`, LAMMPS also provides a plugin mode, allowing one to build LAMMPS and a plugin separately.

Now download the LAMMPS code (`8Apr2021` or later), and uncompress it:
```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_23Jun2022.tar.gz
tar xf stable_23Jun2022.tar.gz
```

The source code of LAMMPS is stored in directory `lammps-stable_23Jun2022`. The directory of the source code should be specified as the CMAKE argument `LAMMPS_SOURCE_ROOT` during installation of the DeePMD-kit C++ interface. Now go into the LAMMPS directory and create a directory called `build`

```bash
mkdir -p lammps-stable_23Jun2022/build/
cd lammps-stable_23Jun2022/build/
```
Now build LAMMPS. Note that `PLUGIN` and `KSPACE` package must be enabled, and `BUILD_SHARED_LIBS` must be set to `yes`. You can install any other package you want.
```bash
cmake -D PKG_PLUGIN=ON -D PKG_KSPACE=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make -j4
make install
```

If everything works fine, you will end up with an executable `${deepmd_root}/bin/lmp`.
```bash
${deepmd_root}/bin/lmp -h
```
