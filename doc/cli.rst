.. _cli:

Command line interface
======================

DeePMD-kit ``dp`` command
-------------------------

.. argparse::
   :module: deepmd.tf.entrypoints.main
   :func: main_parser
   :prog: dp

DPA-ADAPT command line interface
--------------------------------

The ``dpaad`` command is a short alias for ``dpa-adapt`` and exposes the same
subcommands and options.

.. argparse::
   :module: dpa_adapt.cli
   :func: get_parser
   :prog: dpa-adapt
