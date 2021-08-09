# md
set (MAKE_FF_AD FALSE) 

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
find_package(xdrfile REQUIRED)

list (APPEND MD_INCLUDE_PATH "include")
list (APPEND MD_INCLUDE_PATH ${XDRFILE_INCLUDE_DIRS})
include_directories (${MD_INCLUDE_PATH})

file(GLOB MD_SRC src/*.cc src/*.cpp)
add_library(${LIB_DEEPMD_NATIVE} SHARED ${MD_SRC})

set(MDNN_SOURCE_FILES mdnn.cc)
if (MAKE_FF_AD)
  set(MDAD_SOURCE_FILES mdad.cc)
  set(MDFF_SOURCE_FILES mdff.cc)
endif()

function(_add_md_variant variant_name prec_def)
set (dp_mdnn_name "dp_mdnn${variant_name}")
set (dp_mdff_name "dp_mdff${variant_name}")
set (dp_mdad_name "dp_mdad${variant_name}")

add_executable(${dp_mdnn_name} ${MDNN_SOURCE_FILES})
if (MAKE_FF_AD)
  add_executable(${dp_mdff_name} ${MDFF_SOURCE_FILES})
  add_executable(${dp_mdad_name} ${MDAD_SOURCE_FILES})
endif()

target_link_libraries(${dp_mdnn_name} ${LIB_DEEPMD_NATIVE} ${LIB_DEEPMD_OP} ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
if (MAKE_FF_AD)
  target_link_libraries(${dp_mdad_name} ${LIB_DEEPMD_NATIVE} ${LIB_DEEPMD_OP} ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
  target_link_libraries(${dp_mdff_name} ${LIB_DEEPMD_NATIVE}                  ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
endif()

set_target_properties(
  ${dp_mdnn_name}
  PROPERTIES
  LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
  INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
)
if (MAKE_FF_AD)
  set_target_properties(
    ${dp_mdad_name}
    PROPERTIES
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
    INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
    )
  set_target_properties(
    ${dp_mdff_name}
    PROPERTIES
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
    INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
    )
endif()

install(
  TARGETS	${LIB_DEEPMD_NATIVE}   
  DESTINATION	lib/
  )
install(
  TARGETS	${dp_mdnn_name} 
  DESTINATION	bin/
  )
if (MAKE_FF_AD)
  install(
    TARGETS	${dp_mdad_name} 
    DESTINATION	bin/
    )
  install(
    TARGETS	${dp_mdff_name} 
    DESTINATION	bin/
    )
endif()
endfunction()
_add_md_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_md_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")
