file(GLOB LMP_HEADER *.h)
file(GLOB LMP_SRC *.cpp)
file(GLOB LMP_SHSCRIPT *.sh)

unset (LMP_INSTALL_FILES)
list (APPEND LMP_INSTALL_FILES ${LMP_HEADER})
list (APPEND LMP_INSTALL_FILES ${LMP_SRC})
list (APPEND LMP_INSTALL_FILES ${LMP_SHSCRIPT})

configure_file("env.sh.in" "env.sh" @ONLY)
configure_file("pair_deepmd.h.in" "pair_deepmd.h" @ONLY)
configure_file("lammps_install_list.txt.in" "lammps_install_list.txt" @ONLY)
