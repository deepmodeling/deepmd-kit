# Prepare data with dpdata

One can use a convenient tool [`dpdata`](https://github.com/deepmodeling/dpdata) to convert data directly from the output of first principle packages to the DeePMD-kit format.

To install one can execute

```bash
pip install dpdata
```

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

## Automatic conversion cache

Training and validation data groups can specify a dpdata `format` to convert
external inputs automatically. Converted datasets are stored in
`.deepmd_dpdata_cache` in the working directory and are excluded from recursive
input discovery. The directory must be owned by the current user, must not be a
symbolic link, and on POSIX must have mode `0700`. DeePMD-kit creates new caches
with these permissions and rejects existing shared or untrusted caches before
reading their datasets or manifests. Remove an untrusted cache and rerun the
conversion rather than changing permissions on potentially modified contents.

LMDB datasets use mixed-type batches. In the TensorFlow backend, use `se_atten`
or a hybrid whose components all support mixed-type data. Fixed-composition
descriptors such as `se_e2_a`, `se_e2_r`, and `se_e3` cannot consume LMDB batches;
for these descriptors, set `out_format: deepmd/npy` when converting external
fixed-composition inputs instead. Unsupported descriptor/data combinations
raise an actionable error before statistics or network construction.
