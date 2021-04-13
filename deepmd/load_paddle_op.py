from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

setup(
    name='paddle_ops',
    ext_modules=CppExtension(
        sources=['../source/op/paddle_ops/srcs/pd_prod_env_mat_multi_devices_cpu.cc',
        '../source/op/paddle_ops/srcs/pd_prod_env_mat_multi_devices_cuda.cc',
        '../source/op/paddle_ops/srcs/pd_prod_force_se_a_multi_devices_cpu.cc',
        '../source/op/paddle_ops/srcs/pd_prod_force_se_a_multi_devices_cuda.cc',
        '../source/op/paddle_ops/srcs/pd_prod_virial_se_a_multi_devices_cpu.cc',
        '../source/op/paddle_ops/srcs/pd_prod_virial_se_a_multi_devices_cuda.cc'],
        include_dirs=["../source/lib/include/","/usr/local/cuda-10.1/targets/x86_64-linux/include/"],
        library_dirs=["../build/lib/", "/usr/local/cuda-10.1/lib64"],
        extra_link_args=["-ldeepmd","-lcudart"]
    )
)

