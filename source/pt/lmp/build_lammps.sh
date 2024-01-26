#move this file to $lammps_dir/build and run
cmake ../cmake/ -DPKG_DEEPMD=ON -DPKG_MOLECULE=ON -DDEEPMD_INCLUDE_PATH=../../../deepmd-kit/source/pt/api_cc/include/ -DDEEPMD_LIB_PATH=../../../deepmd-kit/source/pt/api_cc/
make -j10
