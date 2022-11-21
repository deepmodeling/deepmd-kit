cmake_minimum_required(VERSION 3.7)
project(infer_water)

# find DeePMD-kit
find_package(DeePMD REQUIRED)

# C++ example
add_executable(infer_water_cc infer_water.cpp)
# link DeePMD-kit C++ API
target_link_libraries(infer_water_cc PRIVATE DeePMD::deepmd_cc)

# C example
add_executable(infer_water_c infer_water.c)
# link DeePMD-kit C API
target_link_libraries(infer_water_c PRIVATE DeePMD::deepmd_c)

# header-only example
add_executable(infer_water_hpp infer_water_hpp.cpp)
# link DeePMD-kit C API
target_link_libraries(infer_water_hpp PRIVATE DeePMD::deepmd_c)

# nlist
add_executable(infer_water_nlist infer_water_nlist.cpp)
# link DeePMD-kit C API
target_link_libraries(infer_water_nlist PRIVATE DeePMD::deepmd_c)
