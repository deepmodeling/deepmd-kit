# train

configure_file("run_config.ini" "${CMAKE_CURRENT_BINARY_DIR}/run_config.ini" @ONLY)

file(GLOB LIB_PY main.py )

install(
  FILES		${LIB_PY} 
  DESTINATION	deepmd
)
install(
  FILES		${CMAKE_CURRENT_BINARY_DIR}/run_config.ini
  DESTINATION	deepmd/pkg_config
)
