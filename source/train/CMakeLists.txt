# train

file(GLOB LIB_PY  Data.py Model.py Test.py)

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
