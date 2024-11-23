Training Parameters
======================================
.. note::
   One can load, modify, and export the input file by using our effective web-based tool `DP-GUI <https://dpgui.deepmodeling.com/input/deepmd-kit-2.0>`_ online or hosted using the :ref:`command line interface <cli>` :code:`dp gui`. All training parameters below can be set in DP-GUI. By clicking "SAVE JSON", one can download the input file for further training.

.. note::
   One can benefit from IntelliSense and validation when
   :ref:`writing JSON files using Visual Studio Code <json_vscode>`.
   See :ref:`here <json_vscode>` to learn how to configure.

.. dargs::
   :module: deepmd.utils.argcheck
   :func: gen_args

.. _json_vscode:

Writing JSON files using Visual Studio Code
-------------------------------------------

When writing JSON files using `Visual Studio Code <https://code.visualstudio.com/>`_, one can benefit from IntelliSense and
validation by adding a `JSON schema <https://json-schema.org/>`_.
To do so, in a VS Code workspace, one can generate a JSON schema file for the input file by running the following command:

.. code-block:: bash

   dp doc-train-input --out-type json_schema > deepmd.json

Then one can `map the schema <https://code.visualstudio.com/docs/languages/json#_mapping-to-a-schema-in-the-workspace>`_
by updating the workspace settings in the `.vscode/settings.json` file as follows:

.. code-block:: json

   {
      "json.schemas": [
         {
               "fileMatch": [
                  "/**/*.json"
               ],
               "url": "./deepmd.json"
         }
      ]
   }
