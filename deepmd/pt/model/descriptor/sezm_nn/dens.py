# SPDX-License-Identifier: LGPL-3.0-or-later
"""
DeNS-specific SeZM modules.

This module provides the force embedding together with the
parallel SeZM `dens` fitting branches:

1. An energy head operating on the scalar descriptor.
2. A clean-force head operating on the final equivariant latent.
3. A denoising head operating on the same latent.
"""

from __future__ import (
    annotations,
)

import copy
import math
from typing import (
    Any,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.task.sezm_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .so3 import (
    SO3Linear,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)

_SQRT_2 = math.sqrt(2.0)
_SQRT_INV_3 = 1.0 / math.sqrt(3.0)
_SQRT_4PI_OVER_3 = math.sqrt(4.0 * math.pi / 3.0)


def _build_real_sh_norm(lmax: int, *, device: torch.device) -> torch.Tensor:
    """Precompute real-spherical-harmonic normalization factors."""
    norm = torch.zeros(lmax + 1, lmax + 1, dtype=torch.float64, device=device)
    for l in range(lmax + 1):
        for m in range(l + 1):
            norm[l, m] = math.sqrt(
                (2 * l + 1)
                / (4.0 * math.pi)
                * math.exp(math.lgamma(l - m + 1) - math.lgamma(l + m + 1))
            )
    return norm


def _associated_legendre_all(
    lmax: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate associated Legendre polynomials `P_l^m(x)` up to `lmax`.

    Parameters
    ----------
    lmax
        Maximum angular degree.
    x
        Cosine values with shape `(N,)`.

    Returns
    -------
    torch.Tensor
        Tensor with shape `(lmax + 1, lmax + 1, N)` where the second axis is
        `m`. Entries with `m > l` stay zero.
    """
    n_sample = x.shape[0]
    out = x.new_zeros((lmax + 1, lmax + 1, n_sample))
    out[0, 0] = 1.0
    if lmax == 0:
        return out

    sin_theta = torch.sqrt((1.0 - x * x).clamp_min(0.0))
    for m in range(1, lmax + 1):
        out[m, m] = -(2 * m - 1) * sin_theta * out[m - 1, m - 1]
    for m in range(lmax):
        out[m + 1, m] = (2 * m + 1) * x * out[m, m]
    for m in range(lmax + 1):
        for l in range(m + 2, lmax + 1):
            out[l, m] = (
                (2 * l - 1) * x * out[l - 1, m] - (l + m - 1) * out[l - 2, m]
            ) / float(l - m)
    return out


def _real_spherical_harmonics(
    lmax: int,
    unit_vec: torch.Tensor,
    sh_norm: torch.Tensor,
    sqrt_2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute packed real spherical harmonics in the SeZM `(l, m)` layout.

    Parameters
    ----------
    lmax
        Maximum angular degree.
    unit_vec
        Unit vectors with shape `(N, 3)`.

    Returns
    -------
    torch.Tensor
        Packed real spherical harmonics with shape `(N, (lmax + 1) ** 2)`.
    """
    x = unit_vec[:, 0]
    y = unit_vec[:, 1]
    z = unit_vec[:, 2].clamp(-1.0, 1.0)
    phi = torch.atan2(y, x)
    legendre = _associated_legendre_all(lmax, z)

    out = unit_vec.new_zeros((unit_vec.shape[0], (lmax + 1) ** 2))
    for l in range(lmax + 1):
        for m in range(l + 1):
            base = legendre[l, m] * sh_norm[l, m]
            zero_idx = l * l + l
            if m == 0:
                out[:, zero_idx] = base
                continue
            sin_term = torch.sin(float(m) * phi)
            cos_term = torch.cos(float(m) * phi)
            out[:, zero_idx - m] = sqrt_2 * base * sin_term
            out[:, zero_idx + m] = sqrt_2 * base * cos_term
    return out


class ForceEmbedding(torch.nn.Module):
    """
    Embed atom-wise force inputs into the SeZM SO(3) latent space.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree of the receiving backbone state.
    channels
        Number of channels per `(l, m)` coefficient.
    precision
        Module precision.
    mlp_bias
        Whether the final SO(3) projection uses an `l=0` bias.
    trainable
        Whether the projection weights are trainable.
    seed
        Initialization seed.
    eps
        Numerical epsilon used for vector normalization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = True,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE
        self.eps = float(eps)
        self.register_buffer(
            "sqrt_inv_3",
            torch.tensor(_SQRT_INV_3, dtype=self.dtype, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "sqrt_2",
            torch.tensor(_SQRT_2, dtype=self.dtype, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "sh_norm",
            _build_real_sh_norm(self.lmax, device=self.device).to(dtype=self.dtype),
            persistent=True,
        )
        self.proj = SO3Linear(
            lmax=self.lmax,
            in_channels=1,
            out_channels=self.channels,
            n_focus=1,
            dtype=self.dtype,
            mlp_bias=mlp_bias,
            trainable=trainable,
            seed=seed,
        )

    def forward(
        self,
        force_input: torch.Tensor,
        noise_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Project atom-wise force inputs into the SeZM SO(3) layout.

        Parameters
        ----------
        force_input
            Force tensor with shape `(nf, nloc, 3)` or `(N, 3)`.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)` or `(N,)`.
            Only masked atoms contribute non-zero embeddings.

        Returns
        -------
        torch.Tensor
            Force embedding with shape `(nf * nloc, D, 1, channels)`.
        """
        if force_input.ndim == 3:
            force_input = force_input.reshape(-1, 3)
        elif force_input.ndim != 2 or force_input.shape[-1] != 3:
            raise ValueError(
                "`force_input` must have shape (nf, nloc, 3) or (N, 3) for force embedding."
            )

        if noise_mask is None:
            mask = torch.ones(
                force_input.shape[0],
                device=force_input.device,
                dtype=torch.bool,
            )
        else:
            mask = noise_mask.reshape(-1).to(
                dtype=torch.bool, device=force_input.device
            )
            if mask.shape[0] != force_input.shape[0]:
                raise ValueError(
                    "`noise_mask` must match the flattened atom dimension of `force_input`."
                )

        force_input = force_input.to(dtype=self.dtype)
        force_norm = torch.linalg.vector_norm(force_input, dim=-1)
        safe_norm = force_norm.clamp_min(self.eps)
        unit_vec = force_input / safe_norm.unsqueeze(-1)
        sh = _real_spherical_harmonics(
            self.lmax,
            unit_vec,
            self.sh_norm,
            self.sqrt_2,
        )
        sh = sh * (force_norm * self.sqrt_inv_3).unsqueeze(-1)
        sh = sh.view(force_input.shape[0], -1, 1, 1)
        embedded = self.proj(sh)
        return embedded * mask.view(-1, 1, 1, 1).to(dtype=embedded.dtype)


class _SeZMVectorHead(torch.nn.Module):
    """
    Read a Cartesian vector from the `l=1` SeZM latent block.

    Parameters
    ----------
    lmax
        Maximum angular degree of the input latent.
    channels
        Number of input channels per `(l, m)` coefficient.
    precision
        Module precision.
    mlp_bias
        Whether the SO(3) projection uses an `l=0` bias.
    trainable
        Whether parameters are trainable.
    seed
        Initialization seed.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 1:
            raise ValueError("`lmax` must be >= 1 for a vector-valued SeZM head.")
        self.channels = int(channels)
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE
        self.register_buffer(
            "cartesian_scale",
            torch.tensor(_SQRT_4PI_OVER_3, dtype=self.dtype, device=self.device),
            persistent=True,
        )
        self.proj = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=1,
            n_focus=1,
            dtype=self.dtype,
            mlp_bias=mlp_bias,
            trainable=trainable,
            seed=seed,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict Cartesian vectors from the final SeZM equivariant latent.

        Parameters
        ----------
        latent
            Final equivariant latent with shape `(nf * nloc, D, 1, channels)`.

        Returns
        -------
        torch.Tensor
            Cartesian vectors with shape `(nf * nloc, 3)`.
        """
        projected = self.proj(latent.to(dtype=self.dtype))
        l1 = projected[:, 1:4, 0, 0]
        # SeZM keeps the l=1 packed basis as (-y, z, -x), so decode back to
        # Cartesian order (x, y, z) with two sign flips and one permutation.
        return self.cartesian_scale * torch.stack(
            [-l1[:, 2], -l1[:, 0], l1[:, 1]],
            dim=-1,
        )


class SeZMDirectForceHead(_SeZMVectorHead):
    """Predict clean direct forces from the final SeZM latent."""


class SeZMDenoisingHead(_SeZMVectorHead):
    """Predict denoising vectors from the final SeZM latent."""


class SeZMDeNSEnergyHead(SeZMEnergyFittingNet):
    """Energy head used by the SeZM `dens` fitting network."""


class SeZMDeNSFittingNet(torch.nn.Module):
    """
    Parallel SeZM fitting branches for the `dens` mode.

    Parameters
    ----------
    ntypes
        Number of atom types.
    dim_descrpt
        Scalar descriptor width.
    condition_lmax
        Maximum spherical harmonic degree of the descriptor entry state that
        receives the external force embedding.
    latent_lmax
        Maximum spherical harmonic degree of the final equivariant latent.
    channels
        Number of latent channels per `(l, m)` coefficient.
    neuron
        Hidden widths of the scalar energy branch.
    bias_atom_e
        Optional per-type atomic energy bias for the scalar energy branch.
    resnet_dt
        Residual time-step flag for the scalar energy branch.
    numb_fparam
        Number of frame parameters.
    numb_aparam
        Number of atomic parameters.
    dim_case_embd
        Case embedding width for the scalar energy branch.
    case_film_embd
        Whether the scalar energy branch uses case FiLM conditioning.
    activation_function
        Activation function of the scalar energy branch.
    bias_out
        Whether the scalar energy branch uses output bias.
    precision
        Module precision.
    mixed_types
        Whether the scalar energy branch shares parameters across atom types.
    seed
        Initialization seed.
    type_map
        Atom type names.
    default_fparam
        Default frame parameters for the scalar energy branch.
    rcond
        Optional condition number used by the scalar energy branch.
    exclude_types
        Atom types excluded by the scalar energy branch.
    trainable
        Whether the `dens` fitting parameters are trainable.
    atom_ener
        Optional vacuum atomic energy contribution for the scalar energy branch.
    use_aparam_as_mask
        Whether atomic parameters act as masks in the scalar energy branch.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        dim_descrpt: int,
        condition_lmax: int,
        latent_lmax: int,
        channels: int,
        neuron: list[int] | None = None,
        bias_atom_e: torch.Tensor | None = None,
        resnet_dt: bool = False,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        case_film_embd: bool = False,
        activation_function: str = "silu",
        bias_out: bool = False,
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        default_fparam: list[float] | None = None,
        rcond: float | None = None,
        exclude_types: list[int] | None = None,
        trainable: bool | list[bool] = True,
        atom_ener: list[torch.Tensor | None] | None = None,
        use_aparam_as_mask: bool = False,
    ) -> None:
        super().__init__()
        if neuron is None:
            neuron = [0]
        self.ntypes = int(ntypes)
        self.dim_descrpt = int(dim_descrpt)
        self.condition_lmax = int(condition_lmax)
        self.latent_lmax = int(latent_lmax)
        self.channels = int(channels)
        self.neuron = [int(width) for width in neuron]
        self.activation_function = str(activation_function)
        self.precision = str(precision)
        self.mixed_types = bool(mixed_types)
        self.numb_fparam = int(numb_fparam)
        self.numb_aparam = int(numb_aparam)
        self.dim_case_embd = int(dim_case_embd)
        self.case_film_embd = bool(case_film_embd and self.dim_case_embd > 0)
        self.bias_out = bool(bias_out)
        self.resnet_dt = bool(resnet_dt)
        self.type_map = None if type_map is None else list(type_map)
        self.default_fparam = default_fparam
        self.rcond = None if rcond is None else float(rcond)
        self.exclude_types = [] if exclude_types is None else list(exclude_types)
        self.trainable = copy.deepcopy(trainable)
        self.atom_ener = atom_ener
        self.use_aparam_as_mask = bool(use_aparam_as_mask)
        self._return_middle_output = False
        self.has_force_embedding_latent = self.condition_lmax >= 1
        self.has_vector_latent = self.latent_lmax >= 1
        trainable_flag = (
            all(self.trainable)
            if isinstance(self.trainable, list)
            else bool(self.trainable)
        )

        # === Step 1. Build the scalar energy branch ===
        self.energy_head = SeZMDeNSEnergyHead(
            ntypes=self.ntypes,
            dim_descrpt=self.dim_descrpt,
            neuron=self.neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=self.resnet_dt,
            numb_fparam=self.numb_fparam,
            numb_aparam=self.numb_aparam,
            dim_case_embd=self.dim_case_embd,
            case_film_embd=self.case_film_embd,
            activation_function=self.activation_function,
            bias_out=self.bias_out,
            precision=self.precision,
            mixed_types=self.mixed_types,
            seed=child_seed(seed, 0),
            type_map=self.type_map,
            default_fparam=self.default_fparam,
            rcond=self.rcond,
            exclude_types=self.exclude_types,
            trainable=self.trainable,
            atom_ener=self.atom_ener,
            use_aparam_as_mask=self.use_aparam_as_mask,
        )

        # === Step 2. Build force-embedding and vector heads ===
        if self.has_force_embedding_latent:
            self.force_embedding = ForceEmbedding(
                lmax=self.condition_lmax,
                channels=self.channels,
                precision=self.precision,
                mlp_bias=True,
                trainable=trainable_flag,
                seed=child_seed(seed, 1),
            )
        else:
            self.force_embedding = None

        if self.has_vector_latent:
            self.direct_force_head = SeZMDirectForceHead(
                lmax=self.latent_lmax,
                channels=self.channels,
                precision=self.precision,
                mlp_bias=False,
                trainable=trainable_flag,
                seed=child_seed(seed, 2),
            )
            self.denoising_head = SeZMDenoisingHead(
                lmax=self.latent_lmax,
                channels=self.channels,
                precision=self.precision,
                mlp_bias=False,
                trainable=trainable_flag,
                seed=child_seed(seed, 3),
            )
        else:
            self.direct_force_head = None
            self.denoising_head = None

    def output_def(self) -> FittingOutputDef:
        """Return the public fitting output contract for `dens` mode."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "dforce",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def get_dim_fparam(self) -> int:
        """Return the frame-parameter width of the energy branch."""
        return self.energy_head.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        """Return whether the energy branch has default frame parameters."""
        return self.energy_head.has_default_fparam()

    def get_default_fparam(self) -> torch.Tensor | None:
        """Return default frame parameters of the energy branch."""
        return self.energy_head.get_default_fparam()

    def get_dim_aparam(self) -> int:
        """Return the atomic-parameter width of the energy branch."""
        return self.energy_head.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        """Return selected atom types of the energy branch."""
        return self.energy_head.get_sel_type()

    def set_return_middle_output(self, enable: bool) -> None:
        """Enable or disable forwarding of the scalar energy hidden activations."""
        self._return_middle_output = bool(enable)
        self.energy_head.set_return_middle_output(enable)

    def build_force_embedding(
        self,
        force_input: torch.Tensor,
        noise_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the descriptor-entry force embedding from atom-wise force inputs.

        Parameters
        ----------
        force_input
            Force tensor with shape `(nf, nloc, 3)` or `(N, 3)`.
        noise_mask
            Optional corruption mask.

        Returns
        -------
        torch.Tensor
            Force embedding with shape `(nf * nloc, D_cond, 1, channels)`.
        """
        if self.force_embedding is None:
            raise RuntimeError(
                f"SeZM `dens` mode requires descriptor condition_lmax >= 1. Got condition_lmax={self.condition_lmax}."
            )
        return self.force_embedding(force_input, noise_mask=noise_mask)

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Any | None = None,
    ) -> None:
        """
        Update type-related metadata for the scalar energy branch.

        Parameters
        ----------
        type_map
            New atom type map.
        model_with_new_type_stat
            Optional reference model carrying new-type statistics.
        """
        self.type_map = list(type_map)
        ref_energy_head = (
            None
            if model_with_new_type_stat is None
            else model_with_new_type_stat.energy_head
        )
        self.energy_head.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=ref_energy_head,
        )

    def forward(
        self,
        descriptor: torch.Tensor,
        latent: torch.Tensor,
        atype: torch.Tensor,
        *,
        noise_mask: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Run the parallel `dens` fitting branches.

        Parameters
        ----------
        descriptor
            Scalar descriptor with shape `(nf, nloc, dim_descrpt)`.
        latent
            Final equivariant latent with shape `(nf * nloc, D, 1, channels)`.
        atype
            Atom types with shape `(nf, nloc)`.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)`.
        fparam
            Optional frame parameters.
        aparam
            Optional atomic parameters.
        return_components
            If true, also return the clean-force and denoising branches.

        Returns
        -------
        dict[str, torch.Tensor]
            Public outputs contain `energy` and mixed `dforce`.
        """
        if self.direct_force_head is None or self.denoising_head is None:
            raise RuntimeError(
                f"SeZM `dens` mode requires descriptor latent_lmax >= 1. Got latent_lmax={self.latent_lmax}."
            )
        nf, nloc = atype.shape[:2]
        energy_ret = self.energy_head(
            descriptor,
            atype,
            fparam=fparam,
            aparam=aparam,
        )
        clean_force = self.direct_force_head(latent).view(nf, nloc, 3)
        denoising_force = self.denoising_head(latent).view(nf, nloc, 3)

        if noise_mask is None:
            mixed_force = clean_force
        else:
            mask = noise_mask.to(dtype=torch.bool, device=clean_force.device).unsqueeze(
                -1
            )
            mixed_force = torch.where(mask, denoising_force, clean_force)

        result = {
            "energy": energy_ret["energy"],
            "dforce": mixed_force.to(dtype=descriptor.dtype),
        }
        if "middle_output" in energy_ret:
            result["middle_output"] = energy_ret["middle_output"]
        if return_components:
            result["clean_dforce"] = clean_force
            result["denoising_dforce"] = denoising_force
        return result

    def serialize(self) -> dict[str, Any]:
        """Serialize the SeZM `dens` fitting network."""
        state = self.state_dict()
        return {
            "@class": "SeZMDeNSFittingNet",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "dim_descrpt": self.dim_descrpt,
                "condition_lmax": self.condition_lmax,
                "latent_lmax": self.latent_lmax,
                "channels": self.channels,
                "neuron": self.neuron.copy(),
                "resnet_dt": self.resnet_dt,
                "numb_fparam": self.numb_fparam,
                "numb_aparam": self.numb_aparam,
                "dim_case_embd": self.dim_case_embd,
                "case_film_embd": self.case_film_embd,
                "activation_function": self.activation_function,
                "bias_out": self.bias_out,
                "precision": self.precision,
                "mixed_types": self.mixed_types,
                "type_map": self.type_map,
                "default_fparam": self.default_fparam,
                "rcond": self.rcond,
                "exclude_types": self.exclude_types.copy(),
                "trainable": self.trainable,
                "atom_ener": self.atom_ener,
                "use_aparam_as_mask": self.use_aparam_as_mask,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMDeNSFittingNet:
        """Deserialize the SeZM `dens` fitting network."""
        data = data.copy()
        if data.pop("@class") != "SeZMDeNSFittingNet":
            raise ValueError("Invalid class for SeZMDeNSFittingNet deserialization.")
        version = int(data.pop("@version", 1))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        state = {key: safe_numpy_to_tensor(value) for key, value in variables.items()}
        obj.load_state_dict(state)
        return obj
