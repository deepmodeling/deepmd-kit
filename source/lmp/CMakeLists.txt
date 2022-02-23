add_subdirectory(plugin)
if (NOT DEFINED LAMMPS_VERSION_NUMBER)
    # set the default to stable_29Sep2021
    set(LAMMPS_VERSION_NUMBER 20210929)
endif()
message(STATUS "LAMMPS version is ${LAMMPS_VERSION_NUMBER}")

file(GLOB LMP_HEADER *.h)
file(GLOB LMP_SRC *.cpp)
file(GLOB LMP_SHSCRIPT *.sh)

unset (LMP_INSTALL_FILES)
list (APPEND LMP_INSTALL_FILES ${LMP_HEADER})
list (APPEND LMP_INSTALL_FILES ${LMP_SRC})
list (APPEND LMP_INSTALL_FILES ${LMP_SHSCRIPT})

function(_add_lmp_variant variant_name prec_def)
configure_file("env.sh.in" "env${variant_name}.sh" @ONLY)
endfunction()
_add_lmp_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
_add_lmp_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")

configure_file("pair_deepmd.h.in" "pair_deepmd.h" @ONLY)
configure_file("lammps_install_list.txt.in" "lammps_install_list.txt" @ONLY)
