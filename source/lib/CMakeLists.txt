# libmd

set (libname ${LIB_DEEPMD})

file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB INC_SRC include/*.h)
# include_directories ("include/")

add_library(${libname} SHARED ${LIB_SRC})

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)
