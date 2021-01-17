# train

file(GLOB LIB_PY  descrpt_*.py deep_*.py network.py fitting.py learning_rate.py loss.py tab_inter.py ewald_recp.py data_modifier.py data.py data_system.py)

install(
  FILES		${LIB_PY} 
  DESTINATION	deepmd
)
