This directory contains property-fitting training inputs on the QM9 subset in
`../data`:

- `input_torch.json`: DPA1 descriptor.
- `input_dpa4.json`: DPA4/SeZM descriptor.

Some explanations of the shared property parameters:

1. `fitting_net/property_name` is the name of the property to be predicted. It should be consistent with the property name in the dataset. In each system, code will read `set.*/{property_name}.npy` file as prediction label if you use NumPy format data.
1. `fitting_net/task_dim` is the dimension of model output. It should be consistent with the property dimension in the dataset, which means if the shape of data stored in `set.*/{property_name}.npy` is `batch size * 3`, `fitting_net/task_dim` should be set to 3.
1. `fitting/intensive` indicates whether the fitting property is intensive. If `intensive` is `true`, the model output is the average of the property contribution of each atom. If `intensive` is `false`, the model output is the sum of the property contribution of each atom.
