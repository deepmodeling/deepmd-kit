# test libdeepmd_c.tar.gz works with gcc 4.9.0, glibc 2.19
set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))

# assume libdeepmd_c.tar.gz has been created

docker run --rm -v ${SCRIPT_PATH}/../..:/root/deepmd-kit -w /root/deepmd-kit \
	gcc:4.9 \
	/bin/sh -c "tar vxzf libdeepmd_c.tar.gz \
            && cd examples/infer_water \
            && gcc convert_model.c -std=c99 -L ../../libdeepmd_c/lib -I ../../libdeepmd_c/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=../../libdeepmd_c/lib -o convert_model \
            && gcc infer_water.c -std=c99 -L ../../libdeepmd_c/lib -I ../../libdeepmd_c/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=../../libdeepmd_c/lib -o infer_water \
            && ./convert_model \
            && ./infer_water"
