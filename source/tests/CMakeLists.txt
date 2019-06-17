file(GLOB LIB_PY test_descrpt_nonsmth.py test_descrpt_smooth.py test_tab_nonsmth.py test_tab_smooth.py common.py)

install(
  FILES		${LIB_PY}
  DESTINATION	deepmd/tests
)

