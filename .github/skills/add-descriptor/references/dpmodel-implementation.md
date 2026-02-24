# dpmodel Implementation Details

## Required methods

| Method                                                  | Purpose                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| `__init__(self, rcut, rcut_smth, sel, ...)`             | Initialize cutoff, sel, networks, statistics                 |
| `call(self, coord_ext, atype_ext, nlist, mapping=None)` | Forward pass, returns `(descriptor, rot_mat, g2, h2, sw)`    |
| `serialize(self) -> dict`                               | Save to dict with `@class`, `type`, `@version`, `@variables` |
| `deserialize(cls, data) -> Self`                        | Reconstruct from dict                                        |
| `get_rcut() -> float`                                   | Cutoff radius                                                |
| `get_rcut_smth() -> float`                              | Smooth cutoff                                                |
| `get_sel() -> list[int]`                                | Neighbor selection per type                                  |
| `get_ntypes() -> int`                                   | Number of atom types                                         |
| `get_type_map() -> list[str]`                           | Type map                                                     |
| `get_dim_out() -> int`                                  | Output descriptor dimension                                  |
| `get_dim_emb() -> int`                                  | Embedding dimension                                          |
| `get_env_protection() -> float`                         | Environment protection value                                 |
| `mixed_types() -> bool`                                 | Whether descriptor mixes types                               |
| `has_message_passing() -> bool`                         | Whether it uses message passing                              |
| `need_sorted_nlist_for_lower() -> bool`                 | Whether nlist must be sorted                                 |
| `compute_input_stats(merged, path)`                     | Compute davg/dstd from data                                  |
| `set_stat_mean_and_stddev(mean, stddev)`                | Set statistics                                               |
| `get_stat_mean_and_stddev()`                            | Get statistics                                               |
| `change_type_map(type_map, ...)`                        | Handle type map changes                                      |
| `share_params(base_class, shared_level, resume)`        | Parameter sharing                                            |
| `update_sel(cls, train_data, type_map, local_jdata)`    | Auto-update sel                                              |

## Statistics handling

Support both naming conventions via `__getitem__`/`__setitem__`:

```python
def __setitem__(self, key, value):
    if key in ("avg", "data_avg", "davg"):
        self.davg = value
    elif key in ("std", "data_std", "dstd"):
        self.dstd = value
    else:
        raise KeyError(key)


def __getitem__(self, key):
    if key in ("avg", "data_avg", "davg"):
        return self.davg
    elif key in ("std", "data_std", "dstd"):
        return self.dstd
    else:
        raise KeyError(key)
```

## Key utilities

| Utility             | Import from                         | Purpose                        |
| ------------------- | ----------------------------------- | ------------------------------ |
| `EnvMat`            | `deepmd.dpmodel.utils.env_mat`      | Environment matrix computation |
| `EmbeddingNet`      | `deepmd.dpmodel.utils.network`      | Embedding neural network       |
| `NetworkCollection` | `deepmd.dpmodel.utils.network`      | Manages type-indexed networks  |
| `PairExcludeMask`   | `deepmd.dpmodel.utils.exclude_mask` | Type exclusion pairs           |
| `EnvMatStatSe`      | `deepmd.dpmodel.utils.env_mat_stat` | Statistics computation         |

## Array API compatibility (CRITICAL)

All dpmodel code must use `array_api_compat` to work across numpy/torch/jax/paddle:

```python
import array_api_compat

xp = array_api_compat.array_namespace(coord_ext)
device = array_api_compat.device(coord_ext)
```

To check whether a method is within the [array API standard](https://data-apis.org/array-api/), use the following command (query `zeros_like` for example):

```sh
uvx --from array-api-strict python -c "import array_api_strict,pydoc;print(pydoc.render_doc(array_api_strict.zeros_like))"
```

If the method exists, its doc will be printed; otherwise, `AttributeError` is thrown.

For methods of an `Array` class, call (query `Array.shape` for example):

```sh
uvx --from array-api-strict python -c "import array_api_strict,pydoc;print(pydoc.render_doc(array_api_strict._array_object.Array.shape))"
```

Rules:

1. **Never use `np.einsum` on arrays that might be torch tensors** — torch disables `__array_function__` so `np.einsum` fails on tensors with `requires_grad=True`. Use `xp.sum` with broadcasting:

   ```python
   # BAD:  np.einsum("lni,lnj->lij", gg, tr)
   # GOOD: xp.sum(gg[:, :, :, None] * tr[:, :, None, :], axis=1)
   ```

1. **`xp.zeros`/`xp.ones` must include `device=`** — omitting device can trigger CUDA init or create tensors on wrong device:

   ```python
   # BAD:  xp.zeros([2, 1], dtype=nlist.dtype)
   # GOOD: xp.zeros([2, 1], dtype=nlist.dtype, device=array_api_compat.device(nlist))
   ```

1. **`xp.split` with `axis=` keyword doesn't work for torch** — use slicing:

   ```python
   # BAD:  g2, h2 = xp.split(dmatrix, [1], axis=-1)
   # GOOD: g2, h2 = dmatrix[..., :1], dmatrix[..., 1:]
   ```

1. **`xp_take_along_axis` indices must be int64 for torch**.

1. **Don't maintain separate ArrayAPI subclasses** — dpmodel classes should be array_api compatible directly.

1. **Boolean fancy indexing (`arr[mask]`) is not array-API compatible** — use mask multiplication:

   ```python
   # BAD:  gr[ti_mask] += gr_tmp
   # GOOD: gr += gr_tmp * xp.astype(mask[:, None, None], gr_tmp.dtype)
   ```
