# DeePMD model artifacts for inference

Read this reference when the model is a training checkpoint, its extension is
`.pt2`, or the correct backend/export path is unclear.

## Identify the artifact

A suffix identifies a serialization/backend route, not necessarily a model
family. In particular, both DPA3 and DPA4 training checkpoints use `.pt`. Never
classify a `.pt` checkpoint from its filename alone. Inspect its stored model
configuration when needed:

```bash
dp --pt show model.pt descriptor fitting-net type-map
```

| Artifact | Typical role                             | Inference guidance                                                                                                                                |
| -------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.pb`    | TensorFlow frozen model                  | Load with `DeepPot` or use `dp test`.                                                                                                             |
| `.pth`   | Conventional PyTorch frozen model        | Load with `DeepPot` or use `dp test`.                                                                                                             |
| `.pt`    | PyTorch training checkpoint              | Inspect before use. DPA4 supports eager Python evaluation and embedding extraction from a checkpoint; deployment normally uses a frozen artifact. |
| `.pt2`   | DPA4/SeZM AOTInductor deployment archive | Load with `DeepPot`, use `dp test`, or deploy to a compatible LAMMPS build.                                                                       |

Backend selection for inference is normally determined from the model artifact.
Do not add a backend flag merely from the assumed model family.

## DPA4/SeZM

DPA4/SeZM supports Python evaluation from its `.pt` checkpoint, but a `.pt2`
archive is the normal frozen deployment artifact. Freeze with:

```bash
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

The command writes `frozen_model.pt2` for a detected DPA4/SeZM checkpoint.
Evaluate the archive with:

```python
from deepmd.infer import DeepPot

model = DeepPot("frozen_model.pt2")
energy, force, virial = model.eval(coord, cell, atype)
```

For labeled data:

```bash
dp test -m frozen_model.pt2 -s /path/to/system -n 30
```

Check that `atype` follows the model `type_map` and that coordinates/cells use
the units and shapes documented by `DeepPot`.

## Descriptors and DPA4 embeddings

Generic descriptor evaluation remains available through `eval_descriptor` or:

```bash
dp eval-desc -m frozen_model.pt2 -s /path/to/system -o desc_output
```

DPA4 additionally exposes model embeddings from a training checkpoint:

```bash
dp embed -m model.ckpt.pt -s /path/to/system -o embedding.hdf5
```

Do not substitute a `.pt2` archive in the documented `dp embed` checkpoint
workflow without checking the installed command's support.

## Validation

- Confirm that the artifact exists and can be loaded in the target environment.
- Inspect the stored descriptor when `.pt` could mean DPA3 or DPA4.
- Confirm the type map before constructing `atype`.
- Run a small finite energy/force/virial evaluation before a large batch.
- Validate an exported `.pt2` where it will be deployed; the suffix alone does
  not guarantee environment compatibility.

## References

- [DPA4 export, inference, and embeddings](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
- [Python inference](https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/python.html)
- [Show model information](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/show-model-info.html)
