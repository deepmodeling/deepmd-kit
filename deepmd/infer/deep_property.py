# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)

from .deep_eval import (
    DeepEval,
)


class DeepProperty(DeepEval):
    """Properties of structures.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    *args : list
        Positional arguments.
    auto_batch_size : bool or int or AutoBatchSize, default: True
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    neighbor_list : ase.neighborlist.NewPrimitiveNeighborList, optional
        The ASE neighbor list class to produce the neighbor list. If None, the
        neighbor list will be built natively in the model.
    **kwargs : dict
        Keyword arguments.
    """

    def output_def(self) -> ModelOutputDef:
        """
        Get the output definition of this model.
        But in property_fitting, the output definition is not known until the model is loaded.
        So we need to rewrite the output definition after the model is loaded.
        See detail in change_output_def.
        """
        pass

    def change_output_def(self) -> None:
        """
        Change the output definition of this model.
        In property_fitting, the output definition is known after the model is loaded.
        We need to rewrite the output definition and related information.
        """
        self.output_def = ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        self.get_var_name(),
                        shape=[self.get_task_dim()],
                        reducible=True,
                        atomic=True,
                        intensive=self.get_intensive(),
                    ),
                ]
            )
        )
        self.deep_eval.output_def = self.output_def
        self.deep_eval._OUTDEF_DP2BACKEND[self.get_var_name()] = (
            f"atom_{self.get_var_name()}"
        )
        self.deep_eval._OUTDEF_DP2BACKEND[f"{self.get_var_name()}_redu"] = (
            self.get_var_name()
        )

    @property
    def task_dim(self) -> int:
        """Get the output dimension."""
        return self.get_task_dim()

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate properties. If atomic is True, also return atomic property.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates of the atoms, in shape (nframes, natoms, 3).
        cells : np.ndarray
            The cell vectors of the system, in shape (nframes, 9). If the system
            is not periodic, set it to None.
        atom_types : list[int] or np.ndarray
            The types of the atoms. If mixed_type is False, the shape is (natoms,);
            otherwise, the shape is (nframes, natoms).
        atomic : bool, optional
            Whether to return atomic property, by default False.
        fparam : np.ndarray, optional
            The frame parameters, by default None.
        aparam : np.ndarray, optional
            The atomic parameters, by default None.
        mixed_type : bool, optional
            Whether the atom_types is mixed type, by default False.
        **kwargs : dict[str, Any]
            Keyword arguments.

        Returns
        -------
        property
            The properties of the system, in shape (nframes, num_tasks).
        """
        self.change_output_def()
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)
        results = self.deep_eval.eval(
            coords,
            cells,
            atom_types,
            atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        atomic_property = results[self.get_var_name()].reshape(
            nframes, natoms, self.get_task_dim()
        )
        property = results[f"{self.get_var_name()}_redu"].reshape(
            nframes, self.get_task_dim()
        )

        if atomic:
            return (
                property,
                atomic_property,
            )
        else:
            return (property,)

    def get_task_dim(self) -> int:
        """Get the output dimension."""
        return self.deep_eval.get_task_dim()

    def get_intensive(self) -> bool:
        """Get whether the property is intensive."""
        return self.deep_eval.get_intensive()

    def get_var_name(self) -> str:
        """Get the name of the fitting property."""
        return self.deep_eval.get_var_name()


__all__ = ["DeepProperty"]
