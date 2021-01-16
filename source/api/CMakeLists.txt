# train

file(GLOB LIB_PY  descrpt_*.py deep_*.py network.py fitting.py)

install(
  FILES		${LIB_PY} 
  DESTINATION	deepmd
)
