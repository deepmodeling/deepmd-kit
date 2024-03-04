# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractclassmethod,
    abstractmethod,
)
from typing import (
    Dict,
    List,
    Optional,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
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

    class BAM(ABC):
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
        def get_type_map(self) -> Optional[List[str]]:
            """Get the type map."""

        def get_ntypes(self) -> int:
            """Get the number of atom types."""
            tmap = self.get_type_map()
            if tmap is not None:
                return len(tmap)
            else:
                raise ValueError(
                    "cannot infer the number of types from a None type map"
                )

        @abstractmethod
        def get_sel(self) -> List[int]:
            """Returns the number of selected atoms for each type."""
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
        def get_sel_type(self) -> List[int]:
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
        def fwd(
            self,
            extended_coord: t_tensor,
            extended_atype: t_tensor,
            nlist: t_tensor,
            mapping: Optional[t_tensor] = None,
            fparam: Optional[t_tensor] = None,
            aparam: Optional[t_tensor] = None,
        ) -> Dict[str, t_tensor]:
            pass

        @abstractmethod
        def serialize(self) -> dict:
            pass

        @abstractclassmethod
        def deserialize(cls):
            pass

        def do_grad_r(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.

            """
            odef = self.fitting_output_def()
            if var_name is None:
                require: List[bool] = []
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
                require: List[bool] = []
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

        def get_model_def_script(self) -> str:
            # TODO: implement this method; saved to model
            raise NotImplementedError

    setattr(BAM, fwd_method_name, BAM.fwd)
    delattr(BAM, "fwd")

    return BAM
