# Prepare data with dpdata

DeePMD-kit includes [`dpdata`](https://github.com/deepmodeling/dpdata) and uses it to convert data from first-principles packages to the DeePMD-kit format.

Labeled `.xyz` and `.extxyz` files are accepted directly by the `systems` key in training and validation configurations. That path is converted automatically; the explicit Python conversion shown below remains useful for other source formats and standalone data preparation.

An example of converting data [VASP](https://www.vasp.at/) data in `OUTCAR` format to DeePMD-kit data can be found at

```
$deepmd_source_dir/examples/data_conv
```

Switch to that directory, then one can convert data by using the following python script

```python
import dpdata

dsys = dpdata.LabeledSystem("OUTCAR")
dsys.to("deepmd/npy", "deepmd_data", set_size=dsys.get_nframes())
```

`get_nframes()` method gets the number of frames in the `OUTCAR`, and the argument `set_size` enforces that the set size is equal to the number of frames in the system, viz. only one `set` is created in the `system`.

The data in DeePMD-kit format is stored in the folder `deepmd_data`.

A list of all [supported data format](https://github.com/deepmodeling/dpdata#load-data) and more nice features of `dpdata` can be found on the [official website](https://github.com/deepmodeling/dpdata).
