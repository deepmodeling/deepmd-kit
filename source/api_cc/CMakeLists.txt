# libmd

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in version.h @ONLY)

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

set(DEEPMD_BACKEND_IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DataModifierTF.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotJAX.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPD.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPT.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPTExpt.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotTF.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepSpinPT.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepSpinPTExpt.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepSpinTF.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepTensorPT.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepTensorTF.cc)
set(DEEPMD_BACKEND_PLUGIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotJAXPlugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPDPlugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPTExptPlugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotPTPlugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DeepPotTFPlugin.cc)
list(REMOVE_ITEM LIB_SRC ${DEEPMD_BACKEND_IMPL_SRC}
     ${DEEPMD_BACKEND_PLUGIN_SRC})

set(libname "${LIB_DEEPMD_CC}")
add_library(${libname} SHARED ${LIB_SRC})

# link: libdeepmd. Backend runtimes are loaded through backend plugins.
target_link_libraries(${libname} PUBLIC ${LIB_DEEPMD})

target_include_directories(
  ${libname}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<INSTALL_INTERFACE:include>)
target_precompile_headers(${libname} PUBLIC [["common.h"]])

if(Protobuf_LIBRARY)
  target_link_libraries(${libname} PRIVATE ${Protobuf_LIBRARY})
endif()

set_target_properties(
  ${libname}
  PROPERTIES INSTALL_RPATH "$ORIGIN;${BACKEND_LIBRARY_PATH}"
             INSTALL_RPATH_USE_LINK_PATH TRUE
             BUILD_RPATH "$ORIGIN/../op/tf;$ORIGIN/../op/pt;$ORIGIN/../op/pd")
target_compile_definitions(${libname} PRIVATE TF_PRIVATE)
if(CMAKE_TESTING_ENABLED)
  target_link_libraries(${libname} PRIVATE coverage_config)
endif()
target_compile_features(${libname} PUBLIC cxx_std_11)

if(BUILD_PY_IF)
  set(DEEPMD_API_CC_INSTALL_DESTINATION deepmd/lib/)
else()
  set(DEEPMD_API_CC_INSTALL_DESTINATION lib/)
endif()

function(deepmd_configure_backend_plugin target_name)
  target_link_libraries(${target_name} PRIVATE ${LIB_DEEPMD_CC})
  target_include_directories(
    ${target_name}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_compile_features(${target_name} PUBLIC cxx_std_11)
  set_target_properties(
    ${target_name}
    PROPERTIES INSTALL_RPATH "$ORIGIN;${BACKEND_LIBRARY_PATH}"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_RPATH "$ORIGIN/../op/tf;$ORIGIN/../op/pt;$ORIGIN/../op/pd")
  if(CMAKE_TESTING_ENABLED)
    target_link_libraries(${target_name} PRIVATE coverage_config)
  endif()
  install(TARGETS ${target_name}
          DESTINATION ${DEEPMD_API_CC_INSTALL_DESTINATION})
  set_property(GLOBAL APPEND PROPERTY DEEPMD_BACKEND_PLUGIN_TARGETS
                                      ${target_name})
endfunction()

if(ENABLE_TENSORFLOW)
  add_library(
    deepmd_backend_tf SHARED
    src/DataModifierTF.cc src/DeepPotTF.cc src/DeepPotTFPlugin.cc
    src/DeepSpinTF.cc src/DeepTensorTF.cc src/common.cc)
  deepmd_configure_backend_plugin(deepmd_backend_tf)
  if(UNIX AND NOT APPLE)
    target_link_libraries(
      deepmd_backend_tf
      PRIVATE -Wl,--no-as-needed TensorFlow::tensorflow_cc -Wl,--as-needed
              TensorFlow::tensorflow_framework)
  else()
    target_link_libraries(
      deepmd_backend_tf PRIVATE TensorFlow::tensorflow_cc
                                TensorFlow::tensorflow_framework)
  endif()
  target_compile_definitions(deepmd_backend_tf PRIVATE BUILD_TENSORFLOW
                                                       TF_PRIVATE)
  if(Protobuf_LIBRARY)
    target_link_libraries(deepmd_backend_tf PRIVATE ${Protobuf_LIBRARY})
  endif()
endif()

if(ENABLE_PYTORCH AND "${OP_CXX_ABI_PT}" EQUAL "${OP_CXX_ABI}")
  add_library(deepmd_backend_pt SHARED src/DeepPotPT.cc src/DeepPotPTPlugin.cc
                                       src/DeepSpinPT.cc src/DeepTensorPT.cc)
  deepmd_configure_backend_plugin(deepmd_backend_pt)
  target_link_libraries(deepmd_backend_pt PRIVATE "${TORCH_LIBRARIES}")
  target_compile_definitions(deepmd_backend_pt PRIVATE BUILD_PYTORCH)

  add_library(
    deepmd_backend_ptexpt SHARED
    src/DeepPotPTExpt.cc src/DeepPotPTExptPlugin.cc src/DeepSpinPTExpt.cc)
  deepmd_configure_backend_plugin(deepmd_backend_ptexpt)
  target_link_libraries(deepmd_backend_ptexpt PRIVATE "${TORCH_LIBRARIES}")
  target_compile_definitions(deepmd_backend_ptexpt PRIVATE BUILD_PYTORCH)
endif()

if(ENABLE_JAX)
  add_library(deepmd_backend_jax SHARED src/DeepPotJAX.cc
                                        src/DeepPotJAXPlugin.cc)
  deepmd_configure_backend_plugin(deepmd_backend_jax)
  target_link_libraries(deepmd_backend_jax PRIVATE TensorFlow::tensorflow_c)
  target_compile_definitions(deepmd_backend_jax PRIVATE BUILD_JAX)
endif()

if(ENABLE_PADDLE AND NOT BUILD_PY_IF)
  add_library(deepmd_backend_pd SHARED src/DeepPotPD.cc src/DeepPotPDPlugin.cc)
  deepmd_configure_backend_plugin(deepmd_backend_pd)
  target_link_libraries(deepmd_backend_pd PRIVATE "${PADDLE_LIBRARIES}")
  target_compile_definitions(deepmd_backend_pd PRIVATE BUILD_PADDLE)
  if(DP_VARIANT STREQUAL "rocm")
    target_link_libraries(deepmd_backend_pd
                          PRIVATE "${hip_LIB_INSTALL_DIR}/libgalaxyhip.so")
  endif()
endif()

if(BUILD_PY_IF)
  install(TARGETS ${libname} DESTINATION deepmd/lib/)
else(BUILD_PY_IF)
  install(
    TARGETS ${libname}
    EXPORT ${CMAKE_PROJECT_NAME}Targets
    DESTINATION lib/)

  install(FILES ${INC_SRC} DESTINATION include/deepmd)

  # make a link to libdeepmd_cc_low.so for compatibility
  install(
    CODE "execute_process( \
COMMAND ${CMAKE_COMMAND} -E create_symlink \
${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${LOW_PREC_VARIANT}${CMAKE_SHARED_LIBRARY_SUFFIX}   \
)")

  if(CMAKE_TESTING_ENABLED)
    add_subdirectory(tests)
  endif()
endif(BUILD_PY_IF)

if(BUILD_TESTING)
  # A compilation test to make sure api_cc can compile without any backend
  add_library(deepmd_cc_test_no_backend SHARED ${LIB_SRC})
  target_link_libraries(deepmd_cc_test_no_backend PUBLIC ${LIB_DEEPMD})
  target_include_directories(
    deepmd_cc_test_no_backend
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:include>)
endif()
