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

add_executable(dp_mdnn ${MDNN_SOURCE_FILES})
if (MAKE_FF_AD)
  add_executable(dp_mdff ${MDFF_SOURCE_FILES})
  add_executable(dp_mdad ${MDAD_SOURCE_FILES})
endif()

target_link_libraries(dp_mdnn ${LIB_DEEPMD_NATIVE} ${LIB_DEEPMD_OP} ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
if (MAKE_FF_AD)
  target_link_libraries(dp_mdad ${LIB_DEEPMD_NATIVE} ${LIB_DEEPMD_OP} ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
  target_link_libraries(dp_mdff ${LIB_DEEPMD_NATIVE}                  ${LIB_DEEPMD} ${XDRFILE_LIBRARIES} ${TensorFlow_LIBRARY})
endif()

set_target_properties(
  dp_mdnn
  PROPERTIES
  LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
  INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
)
if (MAKE_FF_AD)
  set_target_properties(
    dp_mdad
    PROPERTIES
    LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
    INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
    )
  set_target_properties(
    dp_mdff
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
  TARGETS	dp_mdnn 
  DESTINATION	bin/
  )
if (MAKE_FF_AD)
  install(
    TARGETS	dp_mdad 
    DESTINATION	bin/
    )
  install(
    TARGETS	dp_mdff 
    DESTINATION	bin/
    )
endif()
