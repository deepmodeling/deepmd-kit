# Use `dp pretrained` to download built-in models

The `dp pretrained` command provides a simple way to download built-in pre-trained models and store them in a local cache.

## Command syntax

```bash
dp pretrained download <MODEL> [--cache-dir <PATH>]
```

- `<MODEL>`: the built-in model name.
- `--cache-dir <PATH>`: optional cache directory. If omitted, DeePMD-kit uses the default cache path.

## Available built-in models

You can run `dp pretrained download -h` to see the currently supported model list in your installed version.

Examples in this release include:

- `DPA-3.2-5M`
- `DPA-3.1-3M`

## Examples

```bash
# Download to default cache directory
dp pretrained download DPA-3.2-5M

# Download to a custom cache directory
dp pretrained download DPA-3.2-5M --cache-dir ./models
```

The command prints the local path of the downloaded model file on success.

## Use downloaded models via alias

After downloading, you can use the `.pretrained` alias directly in DeepEval/DeepPot workflows.

For example:

```python
from deepmd.infer import DeepPot

# DeePMD-kit resolves this alias to the corresponding local model file
pot = DeepPot("DPA-3.2-5M.pretrained")
```

The `.pretrained` alias is designed for user-facing model selection, while backend details are handled internally.
