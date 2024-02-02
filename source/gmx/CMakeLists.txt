message(STATUS "Build GROMACS plugin")

file(GLOB PATCH_VERSIONS patches/*)
foreach(PATCH_VERSION ${PATCH_VERSIONS})
  configure_file("${PATCH_VERSION}/CMakeLists.txt.patch.in"
                 "${PATCH_VERSION}/CMakeLists.txt.patch" @ONLY)
endforeach(PATCH_VERSION)

set(libgmxname ${LIB_DEEPMD_GROMACS})
file(GLOB LIB_SRC src/*.cpp)
file(GLOB INC_SRC include/*.h)

add_library(${libgmxname} SHARED ${LIB_SRC})
if(DP_USING_C_API)
  target_link_libraries(${libgmxname} PUBLIC ${LIB_DEEPMD_C})
  target_precompile_headers(${libgmxname} PUBLIC [["deepmd.hpp"]])
  remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI})
else()
  target_link_libraries(${libgmxname} PUBLIC ${LIB_DEEPMD_CC})
  target_compile_definitions(${libgmxname} PUBLIC "DP_USE_CXX_API")
endif()
target_compile_definitions(${libgmxname} PRIVATE "DP_GMX_PLUGIN_INTERNAL")
target_include_directories(${libgmxname}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${libgmxname}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/)

set_target_properties(
  ${libgmxname} PROPERTIES INSTALL_RPATH "$ORIGIN;${TensorFlow_LIBRARY_PATH}")

install(
  FILES dp_gmx_patch
  DESTINATION bin
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

install(
  DIRECTORY patches/
  DESTINATION share/deepmd_gromacs_patches
  PATTERN */CMakeLists.txt.patch.in EXCLUDE)

install(FILES ${INC_SRC} DESTINATION include/deepmd)

install(TARGETS ${libgmxname} DESTINATION lib/)
