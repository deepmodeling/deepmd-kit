# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)
from typing import (
    TYPE_CHECKING,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.common import (
    get_hash,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    edge_env_mat,
    graph_from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.utils.env_mat_stat import EnvMatStat as BaseEnvMatStat
from deepmd.utils.env_mat_stat import (
    StatItem,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor import (
        Descriptor,
        DescriptorBlock,
    )


def merge_env_stat(
    base_obj: Union["Descriptor", "DescriptorBlock"],
    link_obj: Union["Descriptor", "DescriptorBlock"],
    model_prob: float = 1.0,
) -> None:
    """Merge descriptor env mat stats from link_obj into base_obj.

    Uses probability-weighted merging: merged = base_stats + link_stats * model_prob,
    where model_prob = link_prob / base_prob.
    Mutates base_obj.stats for chaining (3+ models).

    Parameters
    ----------
    base_obj : Descriptor or DescriptorBlock
        The base descriptor whose stats will be updated.
    link_obj : Descriptor or DescriptorBlock
        The linked descriptor whose stats will be merged in.
    model_prob : float
        The probability weight ratio (link_prob / base_prob).
    """
    if (
        getattr(base_obj, "stats", None) is None
        or getattr(link_obj, "stats", None) is None
    ):
        return
    if getattr(base_obj, "set_stddev_constant", False) and getattr(
        base_obj, "set_davg_zero", False
    ):
        return

    # Weighted merge of StatItem objects
    base_stats = base_obj.stats
    link_stats = link_obj.stats
    merged_stats = {}
    for kk in base_stats:
        merged_stats[kk] = base_stats[kk] + link_stats[kk] * model_prob

    # Compute mean/stddev from merged stats
    base_env = EnvMatStatSe(base_obj)
    base_env.stats = merged_stats
    mean, stddev = base_env()

    # Update base_obj stats for chaining
    base_obj.stats = merged_stats

    # Update buffers in-place: davg/dstd (simple) or mean/stddev (blocks)
    # mean/stddev are numpy arrays; convert to match the buffer's backend
    if hasattr(base_obj, "davg"):
        xp = array_api_compat.array_namespace(base_obj.dstd)
        device = array_api_compat.device(base_obj.dstd)
        if not getattr(base_obj, "set_davg_zero", False):
            base_obj.davg[...] = xp.asarray(
                mean, dtype=base_obj.davg.dtype, device=device
            )
        base_obj.dstd[...] = xp.asarray(
            stddev, dtype=base_obj.dstd.dtype, device=device
        )
    elif hasattr(base_obj, "mean"):
        xp = array_api_compat.array_namespace(base_obj.stddev)
        device = array_api_compat.device(base_obj.stddev)
        if not getattr(base_obj, "set_davg_zero", False):
            base_obj.mean[...] = xp.asarray(
                mean, dtype=base_obj.mean.dtype, device=device
            )
        base_obj.stddev[...] = xp.asarray(
            stddev, dtype=base_obj.stddev.dtype, device=device
        )


class EnvMatStat(BaseEnvMatStat):
    r"""Environment statistics estimating :math:`\mu=\langle R\rangle` and scale."""

    def compute_stat(self, env_mat: dict[str, Array]) -> dict[str, StatItem]:
        """Compute the statistics of the environment matrix for a single system.

        Parameters
        ----------
        env_mat : Array
            The environment matrix.

        Returns
        -------
        dict[str, StatItem]
            The statistics of the environment matrix.
        """
        stats = {}
        for kk, vv in env_mat.items():
            xp = array_api_compat.array_namespace(vv)
            stats[kk] = StatItem(
                number=array_api_compat.size(vv),
                sum=float(xp.sum(vv)),
                squared_sum=float(xp.sum(xp.square(vv))),
            )
        return stats


class EnvMatStatSe(EnvMatStat):
    """Environmental matrix statistics for the se_a/se_r environmental matrix.

    Parameters
    ----------
    descriptor : Descriptor or DescriptorBlock
        The descriptor of the model.
    """

    def __init__(
        self,
        descriptor: Union["Descriptor", "DescriptorBlock"],
        use_graph: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.last_dim = (
            self.descriptor.ndescrpt // self.descriptor.nnei
        )  # se_r=1, se_a=4
        # ``use_graph`` computes the env matrix through the NeighborGraph path
        # (``from_dense_quartet`` -> ``edge_env_mat``) instead of the dense
        # ``EnvMat``, so the input stat runs the SAME machinery the dpa1 graph
        # forward uses. It is BIT-IDENTICAL to the dense path (same neighbor
        # set + padding, ``edge_env_mat`` mirrors ``EnvMat.call``, row-major
        # ``(frame, center, slot)`` edges reshape 1:1 to ``(nf, nloc, nsel)``);
        # only se_a-type (``last_dim == 4``) descriptors may opt in.
        self.use_graph = use_graph

    def _graph_env_mat(
        self,
        extended_coord: Array,
        extended_atype: Array,
        mapping: Array,
        nlist: Array,
    ) -> Array:
        """Env matrix via the NeighborGraph, shaped ``(nf, nloc, nsel, last_dim)``.

        Bit-identical to the dense ``EnvMat.call`` with zero mean / unit std:
        ``from_dense_quartet(compact=False)`` reuses the same neighbor set and
        padding (row-major ``(frame, center, slot)`` edges), ``edge_env_mat``
        mirrors ``EnvMat.call``, and padding / model-excluded edges (already
        ``-1`` in the pre-excluded ``nlist``) carry ``edge_mask=False`` and are
        zeroed -- so the ``(E, 4)`` output reshapes 1:1 back to the dense
        ``(nf, nloc, nsel, 4)`` env-matrix tensor.

        Parameters
        ----------
        extended_coord
            extended coordinates, shape: nf x (nall x 3).
        extended_atype
            extended atom types, shape: nf x nall.
        mapping
            extended-to-local index mapping, shape: nf x nall.
        nlist
            pre-excluded neighbor list, shape: nf x nloc x nsel.

        Returns
        -------
        env_mat
            the environment matrix, shape: nf x nloc x nsel x last_dim.
        """
        xp = array_api_compat.array_namespace(extended_coord, nlist)
        dev = array_api_compat.device(extended_coord)
        nframes, nloc, nsel = nlist.shape
        ntypes = self.descriptor.get_ntypes()
        graph, atype_local = graph_from_dense_quartet(
            extended_coord, extended_atype, nlist, mapping
        )
        # local center type per edge (dst is the local center index)
        center_type = xp.take(atype_local, graph.edge_index[1, :], axis=0)
        zero2 = xp.zeros((ntypes, 4), dtype=graph.edge_vec.dtype, device=dev)
        one2 = xp.ones((ntypes, 4), dtype=graph.edge_vec.dtype, device=dev)
        em = edge_env_mat(
            graph.edge_vec,
            center_type,
            zero2,
            one2,
            self.descriptor.get_rcut(),
            self.descriptor.get_rcut_smth(),
            protection=self.descriptor.get_env_protection(),
            edge_mask=graph.edge_mask,
            return_sw=False,
        )  # (E, 4)
        # zero padding / model-excluded edges (edge_mask=False) so they count
        # as 0 -- exactly like empty slots in the dense path.
        em = em * xp.astype(graph.edge_mask[:, None], em.dtype)
        # row-major (frame, center, slot) -> dense (nf, nloc, nsel, last_dim)
        return xp.reshape(em, (nframes, nloc, nsel, self.last_dim))

    def iter(
        self, data: list[dict[str, np.ndarray | list[tuple[int, int]]]]
    ) -> Iterator[dict[str, StatItem]]:
        """Get the iterator of the environment matrix.

        Parameters
        ----------
        data : list[dict[str, Union[np.ndarray, list[tuple[int, int]]]]]
            The data.

        Yields
        ------
        dict[str, StatItem]
            The statistics of the environment matrix.
        """
        if self.last_dim == 4:
            radial_only = False
        elif self.last_dim == 1:
            radial_only = True
        else:
            raise ValueError(
                "last_dim should be 1 for raial-only or 4 for full descriptor."
            )
        if len(data) == 0:
            # workaround to fix IndexError: list index out of range
            yield from ()
            return
        xp = array_api_compat.array_namespace(data[0]["coord"])
        zero_mean = xp.zeros(
            (
                self.descriptor.get_ntypes(),
                self.descriptor.get_nsel(),
                self.last_dim,
            ),
            dtype=get_xp_precision(xp, "global"),
            device=array_api_compat.device(data[0]["coord"]),
        )
        one_stddev = xp.ones(
            (
                self.descriptor.get_ntypes(),
                self.descriptor.get_nsel(),
                self.last_dim,
            ),
            dtype=get_xp_precision(xp, "global"),
            device=array_api_compat.device(data[0]["coord"]),
        )
        for system in data:
            coord = system["coord"]
            atype = system["atype"]
            box = system.get("box")
            nframes, nloc = atype.shape[:2]
            pair_excl = None
            if "pair_exclude_types" in system:
                pair_excl = PairExcludeMask(
                    self.descriptor.get_ntypes(), system["pair_exclude_types"]
                )
            (
                extended_coord,
                extended_atype,
                mapping,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                coord,
                atype,
                self.descriptor.get_rcut(),
                self.descriptor.get_sel(),
                mixed_types=self.descriptor.mixed_types(),
                box=box,
                # Model-level pair exclusion is a nlist-BUILD transform
                # (decision #18/A4): fold it in here so the input stat matches
                # the model forward, which feeds the descriptor a pre-excluded
                # nlist. Excluded pairs then behave exactly like empty slots
                # (env_mat 0, still counted) -- identical to descriptor-level
                # exclude_types, replacing the previous accumulation-deselect.
                pair_excl=pair_excl,
            )
            if self.use_graph:
                # NeighborGraph env matrix (bit-identical to the dense EnvMat
                # below): the SAME machinery the dpa1 graph forward uses.
                env_mat = self._graph_env_mat(
                    extended_coord, extended_atype, mapping, nlist
                )
            else:
                env_mat_caller = EnvMat(
                    self.descriptor.get_rcut(),
                    self.descriptor.get_rcut_smth(),
                    protection=self.descriptor.get_env_protection(),
                )
                env_mat, _, _ = env_mat_caller.call(
                    extended_coord,
                    extended_atype,
                    nlist,
                    zero_mean,
                    one_stddev,
                    radial_only,
                )
            # apply excluded_types
            exclude_mask = self.descriptor.emask.build_type_exclude_mask(
                nlist, extended_atype
            )
            env_mat *= xp.astype(exclude_mask[..., None], env_mat.dtype)
            # reshape to nframes * nloc at the atom level,
            # so nframes/mixed_type do not matter
            env_mat = xp.reshape(
                env_mat,
                (
                    nframes * nloc,
                    self.descriptor.get_nsel(),
                    self.last_dim,
                ),
            )
            atype = xp.reshape(atype, (nframes * nloc,))
            # (1, nloc) eq (ntypes, 1), so broadcast is possible
            # shape: (ntypes, nloc)
            type_idx = xp.equal(
                xp.reshape(atype, (1, -1)),
                xp.reshape(
                    xp.arange(
                        self.descriptor.get_ntypes(),
                        dtype=xp.int32,
                        device=array_api_compat.device(atype),
                    ),
                    (-1, 1),
                ),
            )
            # NOTE: model-level ``pair_exclude_types`` is NOT re-applied here.
            # It is folded into the neighbor list at BUILD time above
            # (decision #18/A4), so excluded pairs already have env_mat == 0
            # and are counted like empty slots -- the same treatment the model
            # forward gives them.
            for type_i in range(self.descriptor.get_ntypes()):
                dd = env_mat[type_idx[type_i, ...]]
                dd = xp.reshape(
                    dd, (-1, self.last_dim)
                )  # typen_atoms * unmasked_nnei, 4
                env_mats = {}
                env_mats[f"r_{type_i}"] = dd[:, :1]
                if self.last_dim == 4:
                    env_mats[f"a_{type_i}"] = dd[:, 1:]
                yield self.compute_stat(env_mats)

    def get_stat_keys(self) -> list[str]:
        """Get the dataset names required for a complete statistics cache."""
        components = ("r", "a") if self.last_dim == 4 else ("r",)
        return [
            f"{component}_{type_i}"
            for type_i in range(self.descriptor.get_ntypes())
            for component in components
        ]

    def get_hash(self) -> str:
        """Get the hash of the environment matrix.

        Returns
        -------
        str
            The hash of the environment matrix.
        """
        dscpt_type = "se_a" if self.last_dim == 4 else "se_r"
        return get_hash(
            {
                "type": dscpt_type,
                "ntypes": self.descriptor.get_ntypes(),
                "rcut": round(self.descriptor.get_rcut(), 2),
                "rcut_smth": round(self.descriptor.rcut_smth, 2),
                "nsel": self.descriptor.get_nsel(),
                "sel": self.descriptor.get_sel(),
                "mixed_types": self.descriptor.mixed_types(),
            }
        )

    def __call__(self) -> tuple[Array, Array]:
        avgs = self.get_avg()
        stds = self.get_std()

        all_davg = []
        all_dstd = []

        for type_i in range(self.descriptor.get_ntypes()):
            if self.last_dim == 4:
                davgunit = [[avgs[f"r_{type_i}"], 0, 0, 0]]
                dstdunit = [
                    [
                        stds[f"r_{type_i}"],
                        stds[f"a_{type_i}"],
                        stds[f"a_{type_i}"],
                        stds[f"a_{type_i}"],
                    ]
                ]
            elif self.last_dim == 1:
                davgunit = [[avgs[f"r_{type_i}"]]]
                dstdunit = [
                    [
                        stds[f"r_{type_i}"],
                    ]
                ]
            davg = np.tile(davgunit, [self.descriptor.get_nsel(), 1])
            dstd = np.tile(dstdunit, [self.descriptor.get_nsel(), 1])
            all_davg.append(davg)
            all_dstd.append(dstd)

        mean = np.stack(all_davg)
        stddev = np.stack(all_dstd)
        return mean, stddev
