from paddle.utils.cpp_extension import CppExtension, setup
import site

site_package_dir = site.getsitepackages()[0]

setup(
    name='paddle_ops',
    ext_modules=CppExtension(
        sources=['../source/op/paddle_ops/srcs/pd_prod_env_mat_multi_devices_cpu.cc',
        '../source/op/paddle_ops/srcs/pd_prod_force_se_a_multi_devices_cpu.cc',
        '../source/op/paddle_ops/srcs/pd_prod_virial_se_a_multi_devices_cpu.cc'],
        include_dirs=["../source/lib/include/"],
        library_dirs=[site_package_dir+"/deepmd/op"],
        extra_link_args=["-ldeepmd"]
    )
)

