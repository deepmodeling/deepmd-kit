# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data-side transforms of Uni-Mol v1 molecular pretraining.

Ported from Uni-Mol (https://github.com/deepmodeling/Uni-Mol) at commit 90f52c4,
MIT licensed:

    Copyright (c) DP Technology
    This source code is licensed under the MIT license found in the LICENSE
    file in the root directory of that source tree.

Upstream expresses each step as a lazy dataset wrapper
(``unimol/data/*_dataset.py``); here they are plain functions over one frame, so
that a deepmd data loader can call them. The random draws keep upstream's order
and its per-sample seeding, ``hash((seed, epoch, index)) % 1e6``, so a frame
comes out corrupted exactly as upstream would corrupt it.

Corruption has to happen here rather than inside a loss, because the
PyTorch-Exportable backend runs the model before the loss ever sees a frame.

The legacy ``numpy.random`` interface is used on purpose, against deepmd's
usual preference for ``np.random.Generator``: upstream seeds the global
legacy PRNG, and a Generator draws a different stream, which would give
different masks and different noise for the same seed. Every such call is
marked with ``# noqa: NPY002``.
"""

import contextlib
from collections.abc import (
    Callable,
    Iterator,
    Sequence,
)

import numpy as np

__all__ = [
    "add_bos_eos",
    "center_coordinates",
    "crop_atoms",
    "edge_type",
    "make_unimol_data_transform",
    "mask_points",
    "numpy_seed",
    "pair_distance",
    "remove_hydrogen",
    "sample_conformer",
    "unimol_frame_transform",
]


@contextlib.contextmanager
def numpy_seed(seed: int | None, *addl_seeds: int) -> Iterator[None]:
    """Seed the legacy NumPy PRNG, then restore the previous state.

    Mirrors ``unimol/data/data_utils.py:9``. Note the modulus is ``1e6`` here,
    while Uni-Core's own copy uses ``1e8``; the Uni-Mol transforms call this one.
    Hashing a tuple of ints is stable across ``PYTHONHASHSEED``.
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def sample_conformer(
    conformers: Sequence[np.ndarray], seed: int, epoch: int, index: int
) -> np.ndarray:
    """Draw one conformer from the pool.

    Mirrors ``ConformerSampleDataset``. Upstream appends an RDKit 2D conformer to
    the pool while loading; in deepmd that belongs to the offline converter, so
    the pool arrives here already complete and this step needs no RDKit.
    """
    with numpy_seed(seed, epoch, index):
        sample_idx = np.random.randint(len(conformers))  # noqa: NPY002
    return np.asarray(conformers[sample_idx], dtype=np.float32)


def remove_hydrogen(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    remove_hydrogen: bool = False,
    remove_polar_hydrogen: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the hydrogen policy.

    Mirrors ``RemoveHydrogenDataset``. ``remove_hydrogen`` drops every H;
    ``remove_polar_hydrogen`` drops only the trailing run of H, which is what
    Uni-Mol calls polar hydrogens. Upstream maps ``only_polar`` to this pair:
    -1 keeps all, 0 removes all, 1 removes the trailing run.
    """
    atoms = np.asarray(atoms)
    if remove_hydrogen:
        keep = atoms != "H"
        atoms, coordinates = atoms[keep], coordinates[keep]
    if not remove_hydrogen and remove_polar_hydrogen:
        end_idx = 0
        for i, atom in enumerate(atoms[::-1]):
            if atom != "H":
                break
            end_idx = i + 1
        if end_idx != 0:
            atoms, coordinates = atoms[:-end_idx], coordinates[:-end_idx]
    return atoms, coordinates.astype(np.float32)


def crop_atoms(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    seed: int,
    epoch: int,
    index: int,
    max_atoms: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly keep at most ``max_atoms`` atoms.

    Mirrors ``CroppingDataset``. The subset is drawn without replacement and
    without regard to where the atoms are in space.
    """
    if max_atoms and len(atoms) > max_atoms:
        with numpy_seed(seed, epoch, index):
            keep = np.random.choice(len(atoms), max_atoms, replace=False)  # noqa: NPY002
        atoms, coordinates = np.asarray(atoms)[keep], coordinates[keep]
    return atoms, coordinates.astype(np.float32)


def center_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Move the centroid to the origin. Mirrors ``NormalizeDataset``."""
    return (coordinates - coordinates.mean(axis=0)).astype(np.float32)


def mask_points(
    tokens: np.ndarray,
    coordinates: np.ndarray,
    *,
    num_types: int,
    special_indices: Sequence[int],
    pad_idx: int,
    mask_idx: int,
    seed: int,
    epoch: int,
    index: int,
    mask_prob: float = 0.15,
    leave_unmasked_prob: float = 0.05,
    random_token_prob: float = 0.05,
    noise_type: str = "uniform",
    noise: float = 1.0,
) -> dict[str, np.ndarray]:
    """Corrupt a frame the way Uni-Mol pretraining does.

    Mirrors ``MaskPointsDataset.__getitem_cached__``. Of the selected atoms, 90%
    become ``[MASK]``, 5% become a random element and 5% are left alone; all
    three go into the loss targets. Coordinate noise lands on the masked and the
    randomly replaced atoms, not on the ones left alone.

    The order of the random draws matters and is kept: the rounding draw, the
    selection, the two split draws, the noise, then the random elements.

    Returns
    -------
    dict
        ``tokens`` and ``coordinates`` are the corrupted inputs; ``targets``
        holds the true element at every selected position and ``pad_idx``
        elsewhere.
    """
    assert 0.0 < mask_prob < 1.0
    assert 0.0 <= random_token_prob <= 1.0
    assert 0.0 <= leave_unmasked_prob <= 1.0
    assert random_token_prob + leave_unmasked_prob <= 1.0

    weights = None
    if random_token_prob > 0.0:
        weights = np.ones(num_types, dtype=np.float64)
        weights[list(special_indices)] = 0
        weights = weights / weights.sum()

    if noise_type == "trunc_normal":

        def noise_f(n):  # noqa: ANN001, ANN202
            return np.clip(
                np.random.randn(n, 3) * noise,  # noqa: NPY002
                a_min=-noise * 2.0,
                a_max=noise * 2.0,
            )
    elif noise_type == "normal":

        def noise_f(n):  # noqa: ANN001, ANN202
            return np.random.randn(n, 3) * noise  # noqa: NPY002
    elif noise_type == "uniform":

        def noise_f(n):  # noqa: ANN001, ANN202
            return np.random.uniform(low=-noise, high=noise, size=(n, 3))  # noqa: NPY002
    else:

        def noise_f(n):  # noqa: ANN001, ANN202
            return 0.0

    with numpy_seed(seed, epoch, index):
        sz = len(tokens)
        assert sz > 0
        # A random addend rounds the count probabilistically, so a small
        # molecule can end up with no masked atom at all.
        num_mask = int(mask_prob * sz + np.random.rand())  # noqa: NPY002
        mask_idc = np.random.choice(sz, num_mask, replace=False)  # noqa: NPY002
        mask = np.full(sz, False, dtype=bool)
        mask[mask_idc] = True

        targets = np.full(len(mask), pad_idx, dtype=np.int64)
        targets[mask] = np.asarray(tokens)[mask]

        rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)  # noqa: NPY002
            if random_token_prob == 0.0:
                unmask, rand_mask = rand_or_unmask, None
            elif leave_unmasked_prob == 0.0:
                unmask, rand_mask = None, rand_or_unmask
            else:
                unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sz) < unmask_prob  # noqa: NPY002
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        new_tokens = np.copy(np.asarray(tokens))
        new_tokens[mask] = mask_idx

        num_mask = mask.astype(np.int32).sum()
        new_coord = np.copy(coordinates)
        new_coord[mask, :] += noise_f(num_mask)

        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                new_tokens[rand_mask] = np.random.choice(num_types, num_rand, p=weights)  # noqa: NPY002

    return {
        "tokens": new_tokens.astype(np.int64),
        "targets": targets.astype(np.int64),
        "coordinates": new_coord.astype(np.float32),
    }


def add_bos_eos(
    tokens: np.ndarray,
    coordinates: np.ndarray,
    targets: np.ndarray,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap a frame in the two virtual tokens.

    Mirrors ``PrependTokenDataset``/``AppendTokenDataset`` as Uni-Mol calls them
    (``tasks/unimol.py:192-206``). Both sit at the origin, which after centering
    is the centroid; both take ``pad_idx`` as target, so neither enters the loss.
    """
    tokens = np.concatenate([[bos_idx], tokens, [eos_idx]]).astype(np.int64)
    targets = np.concatenate([[pad_idx], targets, [pad_idx]]).astype(np.int64)
    zero = np.zeros((1, 3), dtype=coordinates.dtype)
    coordinates = np.concatenate([zero, coordinates, zero], axis=0)
    return tokens, coordinates, targets


def pair_distance(coordinates: np.ndarray) -> np.ndarray:
    """All-pairs Euclidean distance. Mirrors ``DistanceDataset``."""
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)


def edge_type(tokens: np.ndarray, num_types: int) -> np.ndarray:
    """Ordered element-pair id, ``t_i * num_types + t_j``.

    Mirrors ``EdgeTypeDataset``. Special tokens take part, so the table has
    ``num_types ** 2`` entries.
    """
    tokens = np.asarray(tokens)
    return (tokens[:, None] * num_types + tokens[None, :]).astype(np.int64)


def make_unimol_data_transform(
    type_map: Sequence[str],
    *,
    seed: int = 1,
    mask_token: str = "[MASK]",
    **mask_kwargs: float | str,
) -> Callable[[dict, int], dict]:
    """Build the per-frame transform a deepmd data reader installs.

    The reader hands over one already-converted frame, so the conformer draw,
    the hydrogen policy and the size cap are behind us; what remains is centring
    and the corruption itself. The frame comes back with corrupted coordinates
    and element types, plus the two labels the objective needs. The distance
    target is not stored, because it would cost O(natoms^2) per frame; the loss
    derives it from the clean coordinates.

    The atom count is left alone. Cropping here would not work: a frame's atom
    count and the batch layout are settled before the transform runs, so a
    shorter frame would not match the batch it belongs to. The converter applies
    the size cap instead.

    Upstream draws a fresh corruption for the same molecule in every epoch. The
    transform reproduces that by counting how often it has been handed each
    frame and using that count where upstream uses the epoch number, so a
    molecule is corrupted differently each time it comes round. With more than
    one loader process each keeps its own count, which changes the stream but
    not its statistics.

    Parameters
    ----------
    type_map : Sequence[str]
        Element names of the model. It must contain ``mask_token``, since a
        masked atom has to be expressible as a type.
    seed : int
        Together with the frame index and the visit count, this seeds the
        corruption, the way upstream seeds it per sample and epoch.
    mask_token : str
        Name of the pseudo-element standing for ``[MASK]``.
    **mask_kwargs
        Passed to :func:`mask_points`.

    Returns
    -------
    callable
        ``transform(frame, index) -> frame``, matching the reader's hook.
    """
    from deepmd.dpmodel.descriptor.unimol import (
        unimol_vocabulary,
    )

    vocabulary = unimol_vocabulary()
    token_of = {sym: i for i, sym in enumerate(vocabulary)}
    type_map = list(type_map)
    if mask_token not in type_map:
        raise ValueError(
            f"the model type_map must contain {mask_token!r} for Uni-Mol "
            "pretraining, because masked atoms are carried as a pseudo-element"
        )
    type_index = {sym: i for i, sym in enumerate(type_map)}
    unk = token_of["[UNK]"]
    pad = token_of["[PAD]"]
    type_to_token = np.array(
        [token_of.get(sym, unk) for sym in type_map], dtype=np.int64
    )
    token_to_type = np.array(
        [type_index.get(sym, type_index[mask_token]) for sym in vocabulary],
        dtype=np.int64,
    )
    # Upstream draws a replacement over all 26 of its elements. A model whose
    # type_map covers fewer of them could not express the others, and mapping
    # them onto [MASK] would quietly turn a random-element atom into a masked
    # one, so they are excluded from the draw instead. With the full element
    # set, which is what the example configures, nothing is excluded and the
    # distribution is upstream's.
    if "epoch" in mask_kwargs:
        raise TypeError(
            "the epoch is not fixed at build time: the transform advances it "
            "every time it sees a frame, so that a molecule is corrupted "
            "differently each time it comes round"
        )
    specials = [token_of[s] for s in ("[PAD]", "[CLS]", "[SEP]", "[UNK]", mask_token)]
    inexpressible = [
        i
        for i, sym in enumerate(vocabulary)
        if i not in specials and sym not in type_index
    ]
    excluded = [*specials, *inexpressible]
    visits: dict[int, int] = {}

    def transform(frame: dict, index: int) -> dict:
        epoch = visits.get(index, 0) + 1
        visits[index] = epoch
        coord = np.asarray(frame["coord"], dtype=np.float64).reshape(-1, 3)
        atype = np.asarray(frame["atype"], dtype=np.int64).reshape(-1)
        coord = center_coordinates(coord)
        tokens = type_to_token[atype]

        corrupted = mask_points(
            tokens,
            coord,
            num_types=len(vocabulary),
            special_indices=excluded,
            pad_idx=pad,
            mask_idx=token_of[mask_token],
            seed=seed,
            epoch=epoch,
            index=index,
            **mask_kwargs,
        )
        frame = dict(frame)
        frame["coord"] = corrupted["coordinates"].astype(np.float64)
        frame["atype"] = token_to_type[corrupted["tokens"]]
        # Targets stay in Uni-Mol token space, which is what the element head
        # predicts over; unselected atoms carry the padding id.
        frame["unimol_token_target"] = corrupted["targets"].astype(np.int64)
        frame["unimol_coord_target"] = coord.astype(np.float64)
        frame["find_unimol_token_target"] = np.float32(1.0)
        frame["find_unimol_coord_target"] = np.float32(1.0)
        return frame

    return transform


def unimol_frame_transform(
    atoms: Sequence[str],
    conformers: Sequence[np.ndarray],
    *,
    vocab: dict[str, int],
    num_types: int,
    special_indices: Sequence[int],
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    mask_idx: int,
    unk_idx: int,
    seed: int,
    epoch: int,
    index: int,
    remove_hydrogen_: bool = False,
    remove_polar_hydrogen: bool = False,
    max_atoms: int = 256,
    max_seq_len: int = 512,
    **mask_kwargs: float | str,
) -> dict[str, np.ndarray]:
    """Run the whole Uni-Mol pretraining chain for one molecule.

    Mirrors ``UniMolTask.load_dataset`` (``tasks/unimol.py:140-245``): sample a
    conformer, apply the hydrogen policy, crop, centre, tokenize, corrupt, then
    wrap in BOS/EOS and build the distance matrix and edge types. The inputs are
    built from the corrupted coordinates and the targets from the clean ones.
    """
    coordinates = sample_conformer(conformers, seed, epoch, index)
    atoms_arr = np.asarray(atoms)
    atoms_arr, coordinates = remove_hydrogen(
        atoms_arr, coordinates, remove_hydrogen_, remove_polar_hydrogen
    )
    atoms_arr, coordinates = crop_atoms(
        atoms_arr, coordinates, seed, epoch, index, max_atoms
    )
    coordinates = center_coordinates(coordinates)

    tokens = np.asarray([vocab.get(str(a), unk_idx) for a in atoms_arr], dtype=np.int64)
    assert 0 < len(tokens) < max_seq_len

    corrupted = mask_points(
        tokens,
        coordinates,
        num_types=num_types,
        special_indices=special_indices,
        pad_idx=pad_idx,
        mask_idx=mask_idx,
        seed=seed,
        epoch=epoch,
        index=index,
        **mask_kwargs,
    )

    src_tokens, src_coord, tokens_target = add_bos_eos(
        corrupted["tokens"],
        corrupted["coordinates"],
        corrupted["targets"],
        bos_idx,
        eos_idx,
        pad_idx,
    )
    clean_coord = np.concatenate(
        [
            np.zeros((1, 3), dtype=coordinates.dtype),
            coordinates,
            np.zeros((1, 3), dtype=coordinates.dtype),
        ],
        axis=0,
    )
    return {
        "src_tokens": src_tokens,
        "src_coord": src_coord,
        "src_distance": pair_distance(src_coord),
        "src_edge_type": edge_type(src_tokens, num_types),
        "tokens_target": tokens_target,
        "coord_target": clean_coord,
        "distance_target": pair_distance(clean_coord),
    }
