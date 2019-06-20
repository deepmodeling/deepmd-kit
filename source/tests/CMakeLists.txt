file(GLOB LIB_PY *py)

install(
  FILES		${LIB_PY}
  DESTINATION	deepmd/tests
)

