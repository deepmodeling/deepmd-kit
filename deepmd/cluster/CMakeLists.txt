# train

file(GLOB CLS_PY  Local.py Slurm.py)

install(
  FILES		${CLS_PY} 
  DESTINATION	deepmd/cluster
)

