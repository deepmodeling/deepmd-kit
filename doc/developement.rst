.. _coding conventions:

==================
Coding Conventions
==================

Preface
=======

The aim of these coding standards is to help create a codebase with defined and
consistent coding style that every contributor can get easily familiar with. This
will in enhance code readability as there will be no different coding styles from
different contributors and everything will be documented. Also PR diffs will be smaller
because of unified coding style. Finally static typing will help in hunting down
potential bugs before the code is even run.

Contributed code will not be refused merely because it does not
strictly adhere to these conditions; as long as it's internally
consistent, clean, and correct, it probably will be accepted.  But
don't be surprised if the "offending" code gets fiddled over time to
conform to these conventions.

There are also github actions CI checks for python code style which will annotate the
PR diff for you to see the areas where your code is lacking compared to the set standard.

Rules
=====

The code must be compatible with the oldest supported version of python
which is 3.6

The project follows the generic coding conventions as
specified in the `Style Guide for Python Code`_, `Docstring
Conventions`_ and `Typing Conventions`_ PEPs, clarified and extended as follows:

* Do not use "``*``" imports such as ``from module import *``.  Instead,
  list imports explicitly.

* Use 4 spaces per indentation level.  No tabs.

* No one-liner compound statements (i.e., no ``if x: return``: use two
  lines).

* Maximum line length is 88 characters as recomended by
  `black <https://github.com/psf/black>`_ wich is less strict than 
  `Docstring Conventions`_ suggests.

* Use "StudlyCaps" for class names.

* Use "lowercase" or "lowercase_with_underscores" for function,
  method, variable names and module names. For short names,
  joined lowercase may be used (e.g. "tagname").  Choose what is most
  readable.

* No single-character variable names, except indices in loops
  that encompass a very small number of lines
  (``for i in range(5): ...``).

* Avoid lambda expressions.  Use named functions instead.

* Avoid functional constructs (filter, map, etc.).  Use list
  comprehensions instead.

* Use ``"double quotes"`` for string literals, and ``"""triple double
  quotes"""`` for docstring's. Single quotes are OK for
  something like 

* Use f-strings ``s = f"{x:.2f}"`` instead of old style formating with ``"%f" % x``.
  string format method ``"{x:.2f}".format()`` may be used sparsely where it is more
  convenient than f-strings.
  
Whitespace
==========

Python is not C/C++ so whitespace  should be used sparingly to maintain code readability

* Read the *Whitespace in Expressions and Statements*
  section of PEP8_.

* Avoid `trailing whitespaces`_.

* Do not use excessive whitespace in your expressions and statements.

* You should have blank spaces after commas, colons, and semi-colons if it isn’t
  trailing next to the end of a bracket, brace, or parentheses.
* With any operators you should use a space in on both sides of the operator.

* Colons for slicing are considered a binary operator, and should not have any spaces
  between them.

* You should have parentheses with no space, directly next to the function when calling
  functions ``function()``.

* When indexing or slicing the brackets should be directly next to the collection with
  no space ``collection["index"]``.

* Whitespace used to line up variable values is not recommended.

* Make sure you are consistent with the formats you choose when optional choices are
  available.

  .. code-block:: python
    
    f"something {'this' if x else 'that'}"

.. _Style Guide for Python Code:
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _Docstring Conventions: https://www.python.org/dev/peps/pep-0257/
.. _Typing Conventions: https://www.python.org/dev/peps/pep-0484/
.. _Docutils project: http://docutils.sourceforge.net/docs/dev/policies.html
                      #python-coding-conventions
.. _trailing whitespaces: http://www.gnu.org/software/emacs/manual/html_node/
                          emacs/Useless-Whitespace.html

.. attention::

   Thus spake the Lord: Thou shalt indent with four spaces. No more, no less.
   Four shall be the number of spaces thou shalt indent, and the number of thy
   indenting shall be four. Eight shalt thou not indent, nor either indent thou
   two, excepting that thou then proceed to four. Tabs are right out.

                                          Georg Brandl


General advice
==============

 * Get rid of as many ``break`` and ``continue`` statements as possible.

 * Write short functions. All functions should fit within a standard screen.

 * Use descriptive variable names.

Writing documentation in the code
=================================

Here is an example of how to write good docstrings:

    https://github.com/numpy/numpy/blob/master/doc/example.py

The numpy doctring documentation can be found `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_

It is a good practice to run `pydocstyle <https://github.com/PyCQA/pydocstyle>`_
check on your code or use a text editor that does it automatically):

.. code-block:: bash

    $ pydocstyle filename.py

.. _stylecheck:

Run pycodestyle on your code
============================

It's a good idea to run `pycodestyle <https://github.com/PyCQA/pycodestyle>`_
on your code (or use a text editor that does it automatically):

.. code-block:: bash

    $ pycodestyle filename.py

.. _typing:

Run mypy on your code
=====================

It's a good idea to run `mypy <https://github.com/PyCQA/pycodestyle>`_
on your code (or use a text editor that does it automatically):

.. code-block:: bash

    $ mypy filename.py

.. _docstyle:

Run pydocstyle on your code
===========================

It's a good idea to run `pycodestyle <https://github.com/PyCQA/pycodestyle>`_
on your code (or use a text editor that does it automatically):

.. code-block:: bash

    $ pycodestyle filename.py --max-line-length=88

.. _autoformat:

Run black on your code
======================

Another method of enforcing PEP8_ is using a tool such as
`black <https://github.com/psf/black>`_. These tools tend to be
very effective at cleaning up code, but should be used carefully and code
should be retested after cleaning it. Try:

.. code-block:: bash

  $ black --help
