file(GLOB LIB_PY *py *json)

install(
  FILES		${LIB_PY}
  DESTINATION	deepmd/tests
)

