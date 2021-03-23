# libmd

set (libname ${LIB_DEEPMD})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  version.h
  @ONLY
)

if (USE_CUDA_TOOLKIT)
  include_directories("${CUDA_INCLUDE_DIRS}")
endif()

# file(GLOB LIB_SRC src/*.cc src/*.cpp)
file(GLOB LIB_SRC src/pycommon.cc src/NeighborList.cpp src/NNPAtomMap.cc src/PyCaller.cpp src/PyNNPInter.cc src/SimulationRegion.cpp)
file(GLOB INC_SRC include/*.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(${libname} SHARED ${LIB_SRC})

if (USE_CUDA_TOOLKIT)
  target_link_libraries (${libname} ${CUDA_LIBRARIES})
endif()

# target_link_libraries (${libname} /home/hp200266/u01036/gzq/TensorFlow-2.2.0/lib/libpython3.8.so)
target_link_libraries (${libname} python3)

install(TARGETS ${libname} DESTINATION lib/)

install(
  FILES		${INC_SRC}
  DESTINATION	include/deepmd
)
