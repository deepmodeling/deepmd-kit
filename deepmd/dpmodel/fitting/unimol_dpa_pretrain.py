# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol's pretraining objective on a DPA backbone.

The three heads are the same three Uni-Mol trains: the element of a corrupted
atom, the clean coordinates, and the clean pairwise distances. What changes is
what they read. Uni-Mol's transformer carries a pair representation and wraps
each molecule in two virtual tokens; a DPA backbone carries neither, so the
coordinate head reads the equivariant state and the distance head reads pairs of
atom representations. The element head is unchanged in kind -- it reads the
per-atom representation either way.

The two regularisers Uni-Mol places on its own encoder, on the node norm and on
the pair-delta norm, have no counterpart here: they constrain quantities that
belong to that transformer. The objective leaves them at zero weight.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
    safe_cast_array,
)
from deepmd.dpmodel.descriptor.unimol_nn import (
    MaskLMHead,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.fitting.unimol_dpa_heads import (
    EquivariantCoordHead,
    PairDistanceHead,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseFitting.register("unimol_dpa_pretrain")
class UniMolDPAPretrainFitting(NativeOP, BaseFitting):
    r"""Uni-Mol's self-supervised heads, reading a DPA backbone.

    Parameters
    ----------
    ntypes : int
        Number of element types of the model.
    dim_descrpt : int
        Width of the per-atom representation the backbone produces.
    node_readout_lmax : int
        Degree the backbone reads out, which fixes the shape of the equivariant
        state the coordinate head consumes. Must be at least 1.
    n_token : int
        Size of the Uni-Mol vocabulary, which is the width of the element head.
    max_atoms : int
        Largest molecule accepted. The distance head declares a fixed number of
        columns, so this fixes it; larger frames cannot be expressed.
    dist_hidden : int
        Width of the distance head's hidden layer.
    dist_coverage : str
        Which pairs the distance head covers: ``neighbour`` follows the
        backbone's neighbour list, ``all_pairs`` covers every pair as Uni-Mol
        does. See :class:`PairDistanceHead`.
    activation_function : str
        Activation of the heads.
    mask_token_head, coord_head, dist_head : bool
        Build each head.
    precision : str
        Floating-point precision of the parameters.
    seed : int, optional
        Random seed for initialization.
    type_map : list of str, optional
        Element names, as configured on the model.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        node_readout_lmax: int,
        n_token: int = 31,
        max_atoms: int = 256,
        dist_hidden: int = 64,
        dist_coverage: str = "neighbour",
        activation_function: str = "gelu_erf",
        mask_token_head: bool = True,
        coord_head: bool = True,
        dist_head: bool = True,
        precision: str = "float64",
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        **kwargs: float | str | bool,
    ) -> None:
        del kwargs
        self.type_map = list(type_map) if type_map is not None else None
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.node_readout_lmax = node_readout_lmax
        self.n_token = n_token
        self.max_atoms = max_atoms
        self.dist_hidden = dist_hidden
        self.dist_coverage = dist_coverage
        self.activation_function = activation_function
        self.precision = precision
        self.lm_head = (
            MaskLMHead(
                dim_descrpt,
                n_token,
                activation_function,
                precision,
                child_seed(seed, 0),
            )
            if mask_token_head
            else None
        )
        self.coord_head = (
            EquivariantCoordHead(
                lmax=node_readout_lmax,
                channels=dim_descrpt,
                precision=precision,
                seed=child_seed(seed, 1),
            )
            if coord_head
            else None
        )
        self.dist_head = (
            PairDistanceHead(
                dim_descrpt=dim_descrpt,
                hidden=dist_hidden,
                coverage=dist_coverage,
                activation_function=activation_function,
                precision=precision,
                seed=child_seed(seed, 2),
            )
            if dist_head
            else None
        )

    def get_type_map(self) -> list[str]:
        """Element names, as configured on the model."""
        return self.type_map if self.type_map is not None else []

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: object = None
    ) -> None:
        """Adopt a new element list.

        The element head is indexed by Uni-Mol token rather than by deepmd type,
        so nothing but the recorded names changes.
        """
        del model_with_new_type_stat
        self.type_map = list(type_map)
        self.ntypes = len(type_map)

    def compute_input_stats(self, merged, stat_file_path=None, **kwargs) -> None:  # noqa: ANN001, ANN003
        """No input statistics: the heads read a learned representation."""

    def get_dim_fparam(self) -> int:
        """No frame parameters: the objective reads structure only."""
        return 0

    def get_dim_aparam(self) -> int:
        """No atomic parameters."""
        return 0

    def has_default_fparam(self) -> bool:
        """There are no frame parameters, so there is no default either."""
        return False

    def get_default_fparam(self):  # noqa: ANN201
        """There are no frame parameters."""
        return None

    def get_sel_type(self) -> list[int]:
        """Every element takes part in the objective."""
        return []

    def reinit_exclude(self, exclude_types: list[int] | None = None) -> None:
        """Type exclusion is meaningless here: every atom is predicted."""
        if exclude_types:
            raise NotImplementedError(
                "unimol_dpa_pretrain predicts every atom and does not support "
                "excluded types"
            )

    def set_case_embd(self, case_idx: int) -> None:
        """Case embeddings are a multi-task feature these heads do not use."""
        raise NotImplementedError(
            "unimol_dpa_pretrain does not support case embeddings"
        )

    def output_def(self) -> FittingOutputDef:
        """Declare the head outputs.

        All are per-atom, none reduces to a frame total, and none is
        differentiated with respect to coordinates or cell: this objective
        denoises a structure rather than predicting a potential energy surface.
        """
        variables = []
        if self.lm_head is not None:
            variables.append(
                OutputVariableDef(
                    "token_logits",
                    [self.n_token],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        if self.coord_head is not None:
            variables.append(
                OutputVariableDef(
                    "coord_update",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        if self.dist_head is not None:
            # A fixed column count, because the output machinery indexes this
            # definition for every key the heads return. The frame's own atom
            # count is shorter, so the rows are padded and the mask says so.
            for name in ("pair_dist", "pair_mask"):
                variables.append(
                    OutputVariableDef(
                        name,
                        [self.max_atoms],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                    )
                )
        return FittingOutputDef(variables)

    def call_atoms(self, node_ebd: Array, latent: Array, nlist: Array) -> dict:
        """Run the heads on one frame's worth of backbone output.

        Parameters
        ----------
        node_ebd : Array
            Per-atom representation, shape ``(nf, nloc, dim_descrpt)``.
        latent : Array
            The backbone's equivariant state, shape
            ``(nf * nloc, (lmax + 1) ** 2, 1, dim_descrpt)``.
        nlist : Array
            Neighbour list, shape ``(nf, nloc, nnei)``.

        Returns
        -------
        dict
            ``token_logits`` and ``coord_update`` per atom; ``pair_dist`` and
            the ``pair_mask`` saying which pairs it covers.
        """
        node_ebd = safe_cast_array(node_ebd, "global", self.precision)
        latent = safe_cast_array(latent, "global", self.precision)
        xp = array_api_compat.array_namespace(node_ebd)
        nf, nloc = node_ebd.shape[0], node_ebd.shape[1]
        out = {}
        if self.lm_head is not None:
            out["token_logits"] = self.lm_head(node_ebd)
        if self.coord_head is not None:
            out["coord_update"] = xp.reshape(self.coord_head(latent), (nf, nloc, 3))
        if self.dist_head is not None:
            if nloc > self.max_atoms:
                raise ValueError(
                    f"a frame of {nloc} atoms exceeds max_atoms={self.max_atoms}; "
                    "the distance head declares a fixed width, so larger frames "
                    "cannot be expressed. Convert the data with a matching "
                    "max_atoms, or raise it here"
                )
            dist, mask = self.dist_head(node_ebd, nlist)
            pad = self.max_atoms - nloc
            if pad:
                shape = (nf, nloc, pad)
                dev = array_api_compat.device(node_ebd)
                zeros = xp.zeros(shape, dtype=dist.dtype, device=dev)
                dist = xp.concat([dist, zeros], axis=-1)
                mask = xp.concat(
                    [mask, xp.zeros(shape, dtype=mask.dtype, device=dev)], axis=-1
                )
            out["pair_dist"] = dist
            out["pair_mask"] = mask
        return {
            kk: safe_cast_array(vv, self.precision, "global") for kk, vv in out.items()
        }

    def call(self, descriptor: Array, atype: Array, **kwargs) -> dict[str, Array]:  # noqa: ANN003
        """Not reachable: the heads need the equivariant state and the nlist.

        The pretraining path goes through :meth:`call_atoms`, driven by the
        atomic model, because neither the equivariant state nor the neighbour
        list fits the descriptor's five-tuple.
        """
        raise NotImplementedError(
            "unimol_dpa_pretrain reads the backbone's equivariant state and its "
            "neighbour list; it is driven through call_atoms by the atomic model"
        )

    def serialize(self) -> dict:
        """Serialize the fitting."""
        return {
            "@class": "Fitting",
            "type": "unimol_dpa_pretrain",
            "@version": 1,
            "ntypes": self.ntypes,
            "type_map": self.type_map,
            "dim_descrpt": self.dim_descrpt,
            "node_readout_lmax": self.node_readout_lmax,
            "n_token": self.n_token,
            "max_atoms": self.max_atoms,
            "dist_hidden": self.dist_hidden,
            "dist_coverage": self.dist_coverage,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mask_token_head": self.lm_head is not None,
            "coord_head": self.coord_head is not None,
            "dist_head": self.dist_head is not None,
            "lm_head": None if self.lm_head is None else self.lm_head.serialize(),
            "coord_head_net": None
            if self.coord_head is None
            else self.coord_head.serialize(),
            "dist_head_net": None
            if self.dist_head is None
            else self.dist_head.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "UniMolDPAPretrainFitting":
        """Deserialize the fitting."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        parts = {k: data.pop(k) for k in ("lm_head", "coord_head_net", "dist_head_net")}
        obj = cls(**data)
        if parts["lm_head"] is not None:
            obj.lm_head = MaskLMHead.deserialize(parts["lm_head"])
        if parts["coord_head_net"] is not None:
            obj.coord_head = EquivariantCoordHead.deserialize(parts["coord_head_net"])
        if parts["dist_head_net"] is not None:
            obj.dist_head = PairDistanceHead.deserialize(parts["dist_head_net"])
        return obj
