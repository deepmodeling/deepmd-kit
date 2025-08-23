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

## Evaluate descriptors

The descriptors of a model can be evaluated and saved using `dp eval-desc`. A typical usage of `dp eval-desc` is

```bash
dp eval-desc -m graph.pb -s /path/to/system -o desc
```

where `-m` gives the model file, `-s` the path to the system directory (or `-f` for a datafile containing paths to systems), and `-o` the output directory where descriptor files will be saved. The descriptors for each system will be saved as `.npy` files with the format `desc/(system_name).npy`. Each descriptor file contains a 2D array where each row represents one atom's descriptor (shape: nframes×natoms, ndesc).

Several other command line options can be passed to `dp eval-desc`, which can be checked with

```bash
$ dp eval-desc --help
```

An explanation will be provided

```{program-output} dp eval-desc -h

```
