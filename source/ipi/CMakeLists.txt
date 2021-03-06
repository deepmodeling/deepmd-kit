# md

list (APPEND MD_INCLUDE_PATH "include")
include_directories (${MD_INCLUDE_PATH})

file(GLOB IN_SRC src/*.cc src/*.c)
add_library(${LIB_DEEPMD_IPI} SHARED ${IN_SRC})

set(DRIVER_SOURCE_FILES driver.cc)
add_executable(dp_ipi ${DRIVER_SOURCE_FILES})
target_link_libraries(dp_ipi ${LIB_DEEPMD_IPI} ${LIB_DEEPMD_OP} ${LIB_DEEPMD_CC} ${LIB_DEEPMD} ${TensorFlow_LIBRARY})

set_target_properties(
  dp_ipi
  PROPERTIES
  LINK_FLAGS "-Wl,-rpath,'$ORIGIN'/../lib -Wl,-z,defs"
  INSTALL_RPATH "$ORIGIN/../lib:${TensorFlow_LIBRARY_PATH}"
)

install(
  TARGETS	${LIB_DEEPMD_IPI}	   
  DESTINATION	lib/
)
install(
  TARGETS	dp_ipi
  DESTINATION	bin/
)
