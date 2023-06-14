# test libdeepmd_c.tar.gz works with gcc 4.9.0, glibc 2.19
set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

# assume libdeepmd_c.tar.gz has been created

wget "https://drive.google.com/uc?export=download&id=1xldLhzm4uSkq6iPohSycNWAsWqKAenKX" -O ${SCRIPT_PATH}/../../examples/infer_water/"graph.pb"

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	gcc:4.9 \
	/bin/sh -c "tar vxzf libdeepmd_c.tar.gz \
            && cd examples/infer_water \
            && gcc infer_water.c -std=c99 -L ../../libdeepmd_c/lib -I ../../libdeepmd_c/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=../../libdeepmd_c/lib -o infer_water \
            && ./infer_water"
