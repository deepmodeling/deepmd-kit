# Test a model

The frozen model can be used in many ways. The most straightforward test can be performed using `dp test`. A typical usage of `dp test` is

```bash
dp test -m graph.pb -s /path/to/system -n 30
```

where `-m` gives the tested model, `-s` the path to the tested system and `-n` the number of tested frames. Several other command line options can be passed to `dp test`, which can be checked with

```bash
$ dp test --help
```

An explanation will be provided

```{program-output} dp test -h

```

## Evaluate stress

For periodic systems whose test data contain virial labels (`virial.npy`), `dp test` additionally reports the stress error of energy models.
The stress is derived from the virial as $\sigma = -\Xi / V$, where $\Xi$ is the virial and $V$ is the cell volume, i.e. the tensile-positive convention in units of eV/Å³.
Frames with a singular cell are excluded from the stress error.
When at least one frame has a non-singular cell, the summary of each system, as well as the average over all systems, contains

```
Stress MAE         : ... eV/Å³
Stress RMSE        : ... eV/Å³
```

When `-d`/`--detail-file` is given, the per-frame stress is written to `<detail-prefix>.s.out`, in addition to `.e.out`, `.f.out`, `.v.out`, and the other detail files.
Each row corresponds to one frame and contains 18 columns: the nine components of the reference stress followed by the nine components of the predicted stress, all in eV/Å³:

```
data_sxx data_sxy data_sxz data_syx data_syy data_syz data_szx data_szy data_szz pred_sxx pred_sxy pred_sxz pred_syx pred_syy pred_syz pred_szx pred_szy pred_szz
```

## Evaluate descriptors

The descriptors of a model can be evaluated and saved using `dp eval-desc`. A typical usage of `dp eval-desc` is

```bash
dp eval-desc -m graph.pb -s /path/to/system -o desc
```

where `-m` gives the model file, `-s` the path to the system directory (or `-f` for a datafile containing paths to systems), and `-o` the output directory where descriptor files will be saved. Use `--dtype` to choose the output precision (`fp32`, `fp64`, or `native`; default `native`). The descriptors for each system will be saved as `.npy` files with the format `desc/(system_name).npy`. Each descriptor file contains a 3D array with shape (nframes, natoms, ndesc).

Several other command line options can be passed to `dp eval-desc`, which can be checked with

```bash
$ dp eval-desc --help
```

An explanation will be provided

```{program-output} dp eval-desc -h

```
