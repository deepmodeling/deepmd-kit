# LMDB 不同大小 System 的 Batch 拼接实现

本文记录 `deepmd-kit-lmdb` 当前 PyTorch 训练中，LMDB 数据如何把不同 `nloc` 的 frame/system 拼成一个 batch，并进入 mixed-batch forward。

## 核心思路

- 默认 `mixed_batch=False` 时，不真正混合不同大小的 system：`SameNlocBatchSampler` 先按 `nloc` 分组，每个 batch 内 frame 的原子数相同，然后走普通 `torch.stack` collate。
- `mixed_batch=True` 时，batch 内允许不同 `nloc`。实现方式不是 padding 原始输入，而是把 atom-wise 字段按原子维度展平拼接：
  - `coord`: `[sum(nloc_i), 3]`
  - `atype`: `[sum(nloc_i)]`
  - `force` / `aparam` 等 atom-wise 字段同样按第 0 维 `torch.cat`
  - `energy` / `box` / `fparam` / `virial` 等 frame-wise 字段仍按 frame 维 `torch.stack`
- 额外生成两个索引张量保留 frame 边界：
  - `batch`: `[total_atoms]`，每个原子属于第几个 frame
  - `ptr`: `[nframes + 1]`，前缀和边界，例如 `[0, nloc_0, nloc_0+nloc_1, ...]`

因此，第 `i` 个 frame 的局部原子范围可以通过 `coord[ptr[i]:ptr[i + 1]]` 还原。

## Flat Graph 预处理

当 descriptor 是 DPA3/Repflows 路径时，训练侧会从模型 descriptor 取出 `rcut`、`sel`、`a_rcut`、`a_sel`、`mixed_types`，传给 `make_lmdb_mixed_batch_collate(graph_config)`。

collate 阶段会调用 `build_precomputed_flat_graph(...)`，逐个 frame 做邻居图预处理：

1. 通过 `ptr` 切出单个 frame 的 `coord/atype/box`。
1. 对单个 frame 调用 ghost 扩展和 neighbor list 构建。
1. 用 `extended_offset` 把每个 frame 的扩展原子索引平移到全 batch 的 flat index 空间。
1. 拼接得到：
   - `extended_atype`, `extended_batch`, `extended_image`, `extended_ptr`
   - `mapping`: extended atom -> 原始 flat local atom
   - `central_ext_index`: local atom 在 extended atom 列表里的位置
   - `nlist_ext`, `a_nlist_ext`: 指向 extended atom 的邻居表
   - `nlist`, `a_nlist`: 映射回 local flat atom 的邻居表
   - `nlist_mask`, `a_nlist_mask`, `edge_index`, `angle_index`

`extended_image` 和 `mapping` 在 forward 中用于从原始 `coord/box` 重建可求导的 `extended_coord`，保证 force/virial 的 autograd 仍连接到原始输入。

## 调用关系

```text
Trainer.get_dataloader_and_iter_lmdb
  -> DataLoader(
       dataset=LmdbDataset,
       batch_size=_data.batch_size,
       sampler=RandomSampler/SequentialSampler,
       collate_fn=make_lmdb_mixed_batch_collate(graph_config),
     )
    -> LmdbDataset.__getitem__
      -> LmdbDataReader.__getitem__
         读取单个 frame，并把 coord/atype/force/box/energy 等整理成标准形状
    -> _collate_lmdb_mixed_batch
       atom-wise 字段 torch.cat，frame-wise 字段 torch.stack
       生成 batch 和 ptr
    -> build_precomputed_flat_graph        # graph_config 存在时
       生成 flat graph 相关字段

Trainer.get_data
  -> 检测 batch_data 中是否有 batch/ptr
  -> 把 _FLAT_GRAPH_INPUT_KEYS 加入 input_keys
  -> 将 flat graph 字段移动到 DEVICE

ModelWrapper.forward
  -> input_dict.update(flat graph fields)

EnergyModel.forward
  -> batch 和 ptr 非空时进入 forward_common_flat
    -> forward_common_flat_native
       -> rebuild_extended_coord_from_flat_graph
       -> forward_common_lower_flat
          -> DPAtomicModel.forward_common_atomic_flat
             -> descriptor.forward_flat
                -> DPA3.forward_flat
                   -> Repflows.forward_flat
             -> fitting_net.forward_flat
          -> energy_atomic.index_add_(0, batch, ...) 得到 energy_redu
       -> _compute_derivatives_flat        # 需要 force/virial 时
```

## 关键文件

- `deepmd/pt/utils/lmdb_dataset.py`: LMDB PyTorch dataset 和 mixed-batch collate。
- `deepmd/pt/utils/nlist.py`: flat graph 预计算和 extended coord 重建。
- `deepmd/pt/train/training.py`: mixed-batch DataLoader 创建、flat graph 输入字段搬运。
- `deepmd/pt/train/wrapper.py`: 把 flat graph 字段传入模型。
- `deepmd/pt/model/model/ener_model.py`: 检测 `batch/ptr` 并进入 flat forward。
- `deepmd/pt/model/model/make_model.py`: flat forward、frame-wise energy reduction、导数计算。
- `deepmd/pt/model/atomic_model/dp_atomic_model.py`: flat atomic model forward。
- `deepmd/pt/model/descriptor/dpa3.py` 和 `deepmd/pt/model/descriptor/repflows.py`: 消费预计算 flat graph。

## 小结

当前 mixed-batch 的本质是 **flat concatenation + `batch/ptr` 边界索引 + collate 阶段预计算 flat graph**。这样可以在同一个 batch 中放入不同原子数的 system，同时避免在主模型输入层对 `coord/atype` 做全局 padding。
