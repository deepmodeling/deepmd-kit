# config

configure_file("run_config.ini" "${CMAKE_CURRENT_BINARY_DIR}/run_config.ini"
               @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/run_config.ini __init__.py
        DESTINATION deepmd/lib)
