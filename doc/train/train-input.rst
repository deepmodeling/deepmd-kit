Training Parameters
======================================
.. note::
   One can load, modify, and export the input file by using our effective web-based tool `DP-GUI <https://deepmodeling.org/dpgui/input/deepmd-kit-2.0>`_. All training parameters below can be set in DP-GUI. By clicking "SAVE JSON", one can download the input file for furthur training.
   
   
   Available activation functions for descriptor:
      tanh
      gelu
      relu
      relu6
      softplus
      sigmoid
   You can specify one of the above functions as the activation function in the descriptor's configuration in "input.json".
   For example:
      "activation_function": "sigmoid"

.. include:: ../train-input-auto.rst
