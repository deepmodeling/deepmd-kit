# train

configure_file("RunOptions.py.in" "${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py" @ONLY)

file(GLOB LIB_PY main.py calculator.py Model*.py Trainer.py ${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py transform.py doc.py compress.py)

file(GLOB CLS_PY  Local.py Slurm.py)

install(
  FILES		${LIB_PY} 
  DESTINATION	deepmd
)
install(
  FILES		${CLS_PY} 
  DESTINATION	deepmd/cluster
)
install(
  FILES		train.py test.py
  DESTINATION	deepmd
)
