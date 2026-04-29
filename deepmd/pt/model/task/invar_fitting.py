# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.task.fitting import (
    GeneralFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@GeneralFitting.register("invar")
@fitting_check_output
class InvarFitting(GeneralFitting):
    """Construct a fitting net for energy.

    Parameters
    ----------
    var_name : str
        The atomic property to fit, 'energy', 'dipole', and 'polar'.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out : int
        The output dimension of the fitting net.
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_e : torch.Tensor, optional
        Average energy per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    exclude_types: list[int]
        Atomic contributions of the excluded atom types are set zero.
    atom_ener: list[Optional[torch.Tensor]], optional
        Specifying atomic energy contribution in vacuum.
        The value is a list specifying the bias. the elements can be None or np.array of output shape.
        For example: [None, [2.]] means type 0 is not set, type 1 is set to [2.]
        The `set_davg_zero` key in the descriptor should be set.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    use_aparam_as_mask: bool
        If True, the aparam will not be used in fitting net for embedding.
    default_fparam: list[float], optional
        The default frame parameter. If set, when `fparam.npy` files are not included in the data system,
        this value will be used as the default value for the frame parameter in the fitting net.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: torch.Tensor | None = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: float | None = None,
        seed: int | list[int] | None = None,
        exclude_types: list[int] = [],
        atom_ener: list[torch.Tensor | None] | None = None,
        type_map: list[str] | None = None,
        use_aparam_as_mask: bool = False,
        default_fparam: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.dim_out = dim_out
        self.atom_ener = atom_ener
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            remove_vaccum_contribution=None
            if atom_ener is None or len([x for x in atom_ener if x is not None]) == 0
            else [x is not None for x in atom_ener],
            type_map=type_map,
            use_aparam_as_mask=use_aparam_as_mask,
            default_fparam=default_fparam,
            **kwargs,
        )

    def _net_out_dim(self) -> int:
        """Set the FittingNet output dim."""
        return self.dim_out

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "invar"
        data["dim_out"] = self.dim_out
        data["atom_ener"] = self.atom_ener
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        return super().deserialize(data)

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Based on embedding net output, alculate total energy.

        Args:
        - inputs: Embedding matrix. Its shape is [nframes, natoms[0], self.dim_descrpt].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].

        Returns
        -------
        - `torch.Tensor`: Total energy with shape [nframes, natoms[0]].
        """
        out = self._forward_common(descriptor, atype, gr, g2, h2, fparam, aparam)
        result = {self.var_name: out[self.var_name].to(env.GLOBAL_PT_FLOAT_PRECISION)}
        if "middle_output" in out:
            result.update(
                {
                    "middle_output": out["middle_output"].to(
                        env.GLOBAL_PT_FLOAT_PRECISION
                    )
                }
            )
        return result

    def forward_flat(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with flat batch format.

        Parameters
        ----------
        descriptor : torch.Tensor
            Descriptor [total_atoms, descriptor_dim].
        atype : torch.Tensor
            Atom types [total_atoms].
        batch : torch.Tensor
            Frame assignment [total_atoms].
        ptr : torch.Tensor
            Frame boundaries [nframes + 1].
        gr : torch.Tensor | None
            Rotation matrix [total_atoms, e_dim, 3].
        g2 : torch.Tensor | None
            Edge embedding.
        h2 : torch.Tensor | None
            Pair representation.
        fparam : torch.Tensor | None
            Frame parameters [nframes, ndf].
        aparam : torch.Tensor | None
            Atomic parameters [total_atoms, nda].

        Returns
        -------
        result : dict[str, torch.Tensor]
            Model predictions in flat format.
        """
        print("ENTERING InvarFitting.forward_flat()")
        print(f"  descriptor shape: {descriptor.shape}")
        print(f"  atype shape: {atype.shape}")
        print(f"  batch shape: {batch.shape}")
        print(f"  ptr: {ptr.tolist()}")

        # For now, convert to batch format and call the original forward
        # TODO: Implement true flat format processing
        nframes = ptr.numel() - 1
        total_atoms = batch.shape[0]

        # Find max nloc
        nloc_list = []
        for i in range(nframes):
            nloc_i = (ptr[i + 1] - ptr[i]).item()
            nloc_list.append(nloc_i)
        max_nloc = max(nloc_list)

        # Create batch tensors with padding
        device = descriptor.device
        descriptor_batch = torch.zeros(
            (nframes, max_nloc, descriptor.shape[1]),
            dtype=descriptor.dtype,
            device=device,
        )
        atype_batch = torch.full(
            (nframes, max_nloc), -1, dtype=atype.dtype, device=device
        )
        gr_batch = None
        if gr is not None:
            gr_batch = torch.zeros(
                (nframes, max_nloc, gr.shape[1], gr.shape[2]),
                dtype=gr.dtype,
                device=device,
            )
        aparam_batch = None
        if aparam is not None:
            aparam_batch = torch.zeros(
                (nframes, max_nloc, aparam.shape[1]),
                dtype=aparam.dtype,
                device=device,
            )

        # Fill in the data
        for i in range(nframes):
            start_idx = ptr[i].item()
            end_idx = ptr[i + 1].item()
            nloc_i = end_idx - start_idx

            descriptor_batch[i, :nloc_i] = descriptor[start_idx:end_idx]
            atype_batch[i, :nloc_i] = atype[start_idx:end_idx]
            if gr is not None:
                gr_batch[i, :nloc_i] = gr[start_idx:end_idx]
            if aparam is not None:
                aparam_batch[i, :nloc_i] = aparam[start_idx:end_idx]

        print(f"Packed to batch format:")
        print(f"  descriptor_batch shape: {descriptor_batch.shape}")
        print(f"  atype_batch shape: {atype_batch.shape}")

        # Call original forward with batch format
        result_batch = self.forward(
            descriptor_batch,
            atype_batch,
            gr=gr_batch,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam_batch,
        )

        print(f"After batch forward:")
        for key, val in result_batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key} shape: {val.shape}")

        # Unpack batch format back to flat format
        result_flat = {}
        for key, val in result_batch.items():
            if isinstance(val, torch.Tensor):
                val_list = []
                for i in range(nframes):
                    nloc_i = nloc_list[i]
                    val_list.append(val[i, :nloc_i])
                result_flat[key] = torch.cat(val_list, dim=0)
            else:
                result_flat[key] = val

        print(f"Unpacked to flat format:")
        for key, val in result_flat.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key} shape: {val.shape}")

        return result_flat

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
