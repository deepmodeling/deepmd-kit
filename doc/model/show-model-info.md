# Use `dp show` to show the model information

The `dp show` command is designed to display essential information about a trained model checkpoint or frozen model file.
This utility helps to understand the model's architecture, configuration, and parameter statistics in both single-task and multi-task settings.

## Command Syntax

```bash
dp --pt show <INPUT> <ATTRIBUTES...>
```

- `<INPUT>`: Path to the model checkpoint file or frozen model file.
- `<ATTRIBUTES>`: One or more information categories to display. Supported values are:

  - `model-branch`: Shows available branches for multi-task models.
  - `type-map`: Shows the type mapping used by the model.
  - `descriptor`: Displays the model descriptor parameters.
  - `fitting-net`: Displays parameters of the fitting network.
  - `size`: (Supported Backends: PyTorch and PaddlePaddle) Shows the parameter counts for various components.

## Example Usage

```bash
# For a multi-task model (model.pt)
dp show model.pt model-branch type-map descriptor fitting-net size

# For a single-task frozen model (frozen_model.pth)
dp show frozen_model.pth type-map descriptor fitting-net size
```

## Output Description

Depending on the provided attributes and the model type, the output includes:

- **Model Type**

  - Logs whether the loaded model is a _singletask_ or _multitask_ model.

- **model-branch**

  - _Only available for multitask models._
  - Lists all available model branches and the special `"RANDOM"` branch, which refers to a randomly initialized fitting net.

- **type-map**

  - For multitask models: Shows the type map for each branch.
  - For singletask models: Shows the model's type map.

- **descriptor**

  - For multitask models: Displays the descriptor parameter for each branch.
  - For singletask models: Displays the descriptor parameter.

- **fitting-net**

  - For multitask models: Shows the fitting network parameters for each branch.
  - For singletask models: Shows the fitting network parameters.

- **size**

  - Prints the number of parameters for each component (`descriptor`, `fitting-net`, etc.), as well as the total parameter count.

## Example Output

For a singletask model, the output might look like:

```
This is a singletask model
The type_map is ['O', 'H', 'Au']
The descriptor parameter is {'type': 'se_e2_a', 'sel': [46, 92, 4], 'rcut': 4.0}
The fitting_net parameter is {'neuron': [24, 24, 24], 'resnet_dt': True, 'seed': 1}
Parameter counts:
Parameters in descriptor: 19,350
Parameters in fitting-net: 119,091
Parameters in total: 138,441
```

For a multitask model, if `model-branch` is selected, it will additionally display available branches:

```
This is a multitask model
Available model branches are ['branch1', 'branch2', 'RANDOM'], where 'RANDOM' means using a randomly initialized fitting net.
...
```
