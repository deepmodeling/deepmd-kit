# config

# cmake will treat true, false, on, off, 1, 0 as booleans we hope an easy way to
# check it
if(ENABLE_TENSORFLOW)
  set(ENABLE_TENSORFLOW 1)
else()
  set(ENABLE_TENSORFLOW 0)
endif()

if(ENABLE_PYTORCH)
  set(ENABLE_PYTORCH 1)
else()
  set(ENABLE_PYTORCH 0)
endif()

configure_file("run_config.ini" "${CMAKE_CURRENT_BINARY_DIR}/run_config.ini"
               @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/run_config.ini __init__.py
        DESTINATION deepmd/lib)
