# Install LAMMPS

There are two ways to install LAMMPS: the built-in mode and the plugin mode. The built-in mode builds LAMMPS along with the DeePMD-kit and DeePMD-kit will be loaded automatically when running LAMMPS. The plugin mode builds LAMMPS and a plugin separately, so one need to use `plugin load` command to load the DeePMD-kit's LAMMPS plugin library. 

## Install LAMMPS's DeePMD-kit module (built-in mode)
DeePMD-kit provide module for running MD simulation with LAMMPS. Now make the DeePMD-kit module for LAMMPS.

```bash
cd $deepmd_source_dir/source/build
make lammps
```
DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory. If you need low precision version, move `env_low.sh` to `env.sh` in the directory. Now download the LAMMPS code (`29Oct2020` or later), and uncompress it:
```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_29Sep2021.tar.gz
tar xf stable_29Sep2021.tar.gz
```
The source code of LAMMPS is stored in directory `lammps-stable_29Sep2021`. Now go into the LAMMPS code and copy the DeePMD-kit module like this
```bash
cd lammps-stable_29Sep2021/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
```
Now build LAMMPS
```bash
make yes-kspace
make yes-user-deepmd
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
Starting from `8Apr2021`, LAMMPS also provides a plugin mode, allowing one build LAMMPS and a plugin separately.

Now download the LAMMPS code (`8Apr2021` or later), and uncompress it:
```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_29Sep2021.tar.gz
tar xf stable_29Sep2021.tar.gz
```
The source code of LAMMPS is stored in directory `lammps-stable_29Sep2021`. Now go into the LAMMPS code and create a directory called `build`
```bash
mkdir -p lammps-stable_29Sep2021/build/
cd lammps-stable_29Sep2021/build/
```
Now build LAMMPS. Note that `PLUGIN` and `KSPACE` package must be enabled, and `BUILD_SHARED_LIBS` must be set to `yes`. You can install any other package you want.
```bash
cmake -D PKG_PLUGIN=ON -D PKG_KSPACE=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make -j4
make install
```

If everything works fine, you will end up with an executable `${deepmd_root}/lmp`.
```bash
${deepmd_root}/lmp -h
```
