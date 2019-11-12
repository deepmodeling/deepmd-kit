# train

configure_file("RunOptions.py.in" "${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py" @ONLY)

file(GLOB LIB_PY main.py common.py env.py compat.py calculator.py Network.py Deep*.py Data.py DataSystem.py Model*.py Descrpt*.py Fitting.py Loss.py LearningRate.py Trainer.py TabInter.py ${CMAKE_CURRENT_BINARY_DIR}/RunOptions.py)

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
