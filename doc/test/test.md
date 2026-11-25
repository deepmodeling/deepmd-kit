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

For an energy or spin-energy model with symmetric virial labels, use:

```bash
dp test -m model.pt2 -s /path/to/system --symm-virial
```

The command compares symmetric virials and the corresponding stresses in both
metrics and detail files. Logs and detail headers include `# symm_v=1`; metric
columns and filenames are unchanged. The option is off by default and is not
inferred from the model or its training input. It works with every backend
supported by energy-model testing.

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
