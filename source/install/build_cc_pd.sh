set -ex

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]; then
	INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
DP_VARIANT=cuda DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=0 cmake -D ENABLE_TENSORFLOW=OFF \
	-D ENABLE_IPI=FALSE \
	-D USE_CUDA_TOOLKIT=TRUE \
	-D ENABLE_PYTORCH=OFF \
	-D ENABLE_PADDLE=ON \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D USE_TF_PYTHON_LIBS=FALSE \
	${CUDA_ARGS} \
	-D LAMMPS_VERSION=stable_29Aug2024_update1 \
	..
cmake --build . -j${NPROC}
cmake --install .

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"

cd ${BUILD_TMP_DIR}
make lammps
export LAMMPS_SOURCE_ROOT=${BUILD_TMP_DIR}/_deps/lammps_download-src/
cd ${BUILD_TMP_DIR}/_deps/lammps_download-src/src/
\cp -r ${BUILD_TMP_DIR}/USER-DEEPMD .
make no-kspace
make yes-kspace
make no-extra-fix
make yes-extra-fix
# make no-user-deepmdin
make yes-user-deepmd
make serial -j
# make mpi -j 10

cd ${BUILD_TMP_DIR}/../../examples/water/lmp

echo "START INFERENCE..."

# export FLAGS_prim_all=true
# export FLAGS_enable_pir_in_executor=1
# export FLAGS_prim_enable_dynamic=true
# export FLAGS_use_cinn=true

export PATH=/workspace/hesensen/deepmd_partx/deepmd-kit-tmp/source/build/_deps/lammps_download-src/src:$PATH
CUDA_VISIBLE_DEVICES=0 USE_CUDA_TOOLKIT=1 lmp_serial -in paddle_se_e2_a.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=0 lmp_serial -in paddle_se_e2_a.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=1 lmp_serial -in paddle_dpa1.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=0 lmp_serial -in paddle_dpa1.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=1 lmp_serial -in paddle_dpa2.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=1 lmp_serial -in paddle_se_e2_a.lammps 2>&1 | tee paddle_infer_serial.log
# USE_CUDA_TOOLKIT=1 mpirun --allow-run-as-root -np 2 lmp_mpi -in paddle_dpa2.lammps 2>&1 | tee paddle_infer_mpi.log
