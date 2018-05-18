# train

file(GLOB LIB_PY  Data.py DataSystem.py Model.py Test.py TestNorot.py RunOptions.py)

install(
  FILES		${LIB_PY} 
  DESTINATION	lib/deepmd
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
