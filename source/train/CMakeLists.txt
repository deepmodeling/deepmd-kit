# train

configure_file("RunOptions.py.in" "${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py" @ONLY)

file(GLOB LIB_PY  Data.py DataSystem.py Model.py Test.py TestNorot.py ${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py)

file(GLOB CLS_PY  Local.py Slurm.py)

install(
  FILES		${LIB_PY} 
  DESTINATION	lib/deepmd
)
install(
  FILES		${CLS_PY} 
  DESTINATION	lib/deepmd/cluster
)
install(
  PROGRAMS	train.py 
  DESTINATION	bin/
  RENAME	dp_train
)
install(
  PROGRAMS	test.py  
  DESTINATION	bin/
  RENAME	dp_test
)
