# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Optional,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


def make_base_atomic_model(
    t_tensor,
    fwd_method_name: str = "forward_atomic",
):
    """Make the base class for the atomic model.

    Parameters
    ----------
    t_tensor
        The type of the tensor. used in the type hint.
    fwd_method_name
        Name of the forward method. For dpmodels, it should be "call".
        For torch models, it should be "forward".

    """

    class BAM(ABC, PluginVariant, make_plugin_registry("atomic model")):
        """Base Atomic Model provides the interfaces of an atomic model."""

        @abstractmethod
        def fitting_output_def(self) -> FittingOutputDef:
            """Get the output def of developer implemented atomic models."""
            pass

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model.

            By default it is the same as FittingOutputDef, but it
            allows model level wrapper of the output defined by the developer.

            """
            return self.fitting_output_def()

        @abstractmethod
        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            pass

        @abstractmethod
        def get_type_map(self) -> list[str]:
            """Get the type map."""
            pass

        def get_ntypes(self) -> int:
            """Get the number of atom types."""
            return len(self.get_type_map())

        @abstractmethod
        def get_sel(self) -> list[int]:
            """Returns the number of selected atoms for each type."""
            pass

        @abstractmethod
        def set_case_embd(self, case_idx: int) -> None:
            """
            Set the case embedding of this atomic model by the given case_idx,
            typically concatenated with the output of the descriptor and fed into the fitting net.
            """
            pass

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return sum(self.get_sel())

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.get_nsel()

        @abstractmethod
        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""

        @abstractmethod
        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""

        @abstractmethod
        def get_sel_type(self) -> list[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """

        @abstractmethod
        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """

        @abstractmethod
        def mixed_types(self) -> bool:
            """If true, the model
            1. assumes total number of atoms aligned across frames;
            2. uses a neighbor list that does not distinguish different atomic types.

            If false, the model
            1. assumes total number of atoms of each atom type aligned across frames;
            2. uses a neighbor list that distinguishes different atomic types.

            """
            pass

        @abstractmethod
        def has_message_passing(self) -> bool:
            """Returns whether the descriptor has message passing."""

        @abstractmethod
        def need_sorted_nlist_for_lower(self) -> bool:
            """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""

        @abstractmethod
        def fwd(
            self,
            extended_coord: t_tensor,
            extended_atype: t_tensor,
            nlist: t_tensor,
            mapping: Optional[t_tensor] = None,
            fparam: Optional[t_tensor] = None,
            aparam: Optional[t_tensor] = None,
        ) -> dict[str, t_tensor]:
            pass

        @abstractmethod
        def serialize(self) -> dict:
            pass

        @classmethod
        @abstractmethod
        def deserialize(cls, data: dict):
            pass

        @abstractmethod
        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat=None
        ) -> None:
            pass

        def enable_compression(
            self,
            min_nbor_dist: float,
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
        ) -> None:
            """Call descriptor enable_compression().

            Parameters
            ----------
            min_nbor_dist
                The nearest distance between atoms
            table_extrapolate
                The scale of model extrapolation
            table_stride_1
                The uniform stride of the first table
            table_stride_2
                The uniform stride of the second table
            check_frequency
                The overflow check frequency
            """
            raise NotImplementedError("This atomi model doesn't support compression!")

        def make_atom_mask(
            self,
            atype: t_tensor,
        ) -> t_tensor:
            """The atoms with type < 0 are treated as virtual atoms,
            which serves as place-holders for multi-frame calculations
            with different number of atoms in different frames.

            Parameters
            ----------
            atype
                Atom types. >= 0 for real atoms <0 for virtual atoms.

            Returns
            -------
            mask
                True for real atoms and False for virtual atoms.

            """
            # supposed to be supported by all backends
            return atype >= 0

        def do_grad_r(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.

            """
            odef = self.fitting_output_def()
            if var_name is None:
                require: list[bool] = []
                for vv in odef.keys():
                    require.append(self.do_grad_(vv, "r"))
                return any(require)
            else:
                return self.do_grad_(var_name, "r")

        def do_grad_c(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.

            """
            odef = self.fitting_output_def()
            if var_name is None:
                require: list[bool] = []
                for vv in odef.keys():
                    require.append(self.do_grad_(vv, "c"))
                return any(require)
            else:
                return self.do_grad_(var_name, "c")

        def do_grad_(self, var_name: str, base: str) -> bool:
            """Tell if the output variable `var_name` is differentiable."""
            assert var_name is not None
            assert base in ["c", "r"]
            if base == "c":
                return self.fitting_output_def()[var_name].c_differentiable
            return self.fitting_output_def()[var_name].r_differentiable

    setattr(BAM, fwd_method_name, BAM.fwd)
    delattr(BAM, "fwd")

    return BAM
