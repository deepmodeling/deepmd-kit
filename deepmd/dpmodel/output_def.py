# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
from enum import (
    IntEnum,
)
from typing import (
    Dict,
    List,
    Tuple,
)


def check_shape(
    shape: List[int],
    def_shape: List[int],
):
    """Check if the shape satisfies the defined shape."""
    assert len(shape) == len(def_shape)
    if def_shape[-1] == -1:
        if list(shape[:-1]) != def_shape[:-1]:
            raise ValueError(f"{shape[:-1]} shape not matching def {def_shape[:-1]}")
    else:
        if list(shape) != def_shape:
            raise ValueError(f"{shape} shape not matching def {def_shape}")


def check_var(var, var_def):
    if var_def.atomic:
        # var.shape == [nf, nloc, *var_def.shape]
        if len(var.shape) != len(var_def.shape) + 2:
            raise ValueError(f"{var.shape[2:]} length not matching def {var_def.shape}")
        check_shape(list(var.shape[2:]), var_def.shape)
    else:
        # var.shape == [nf, *var_def.shape]
        if len(var.shape) != len(var_def.shape) + 1:
            raise ValueError(f"{var.shape[1:]} length not matching def {var_def.shape}")
        check_shape(list(var.shape[1:]), var_def.shape)


def model_check_output(cls):
    """Check if the output of the Model is consistent with the definition.

    Two methods are assumed to be provided by the Model:
    1. Model.output_def that gives the output definition.
    2. Model.__call__ that defines the forward path of the model.

    """

    @functools.wraps(cls, updated=())
    class wrapper(cls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.md = self.output_def()

        def __call__(
            self,
            *args,
            **kwargs,
        ):
            ret = cls.__call__(self, *args, **kwargs)
            for kk in self.md.keys_outp():
                dd = self.md[kk]
                check_var(ret[kk], dd)
                if dd.reduciable:
                    rk = get_reduce_name(kk)
                    check_var(ret[rk], self.md[rk])
                if dd.r_differentiable:
                    dnr, dnc = get_deriv_name(kk)
                    check_var(ret[dnr], self.md[dnr])
                if dd.c_differentiable:
                    assert dd.r_differentiable
                    check_var(ret[dnc], self.md[dnc])
            return ret

    return wrapper


def fitting_check_output(cls):
    """Check if the output of the Fitting is consistent with the definition.

    Two methods are assumed to be provided by the Fitting:
    1. Fitting.output_def that gives the output definition.
    2. Fitting.__call__ defines the forward path of the fitting.

    """

    @functools.wraps(cls, updated=())
    class wrapper(cls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.md = self.output_def()

        def __call__(
            self,
            *args,
            **kwargs,
        ):
            ret = cls.__call__(self, *args, **kwargs)
            for kk in self.md.keys():
                dd = self.md[kk]
                check_var(ret[kk], dd)
            return ret

    return wrapper


class OutputVariableOperation(IntEnum):
    """Defines the operation of the output variable."""

    _NONE = 0
    """No operation."""
    REDU = 1
    """Reduce the output variable."""
    DERV_R = 2
    """Derivative w.r.t. coordinates."""
    DERV_C = 4
    """Derivative w.r.t. cell."""
    _SEC_DERV_R = 8
    """Second derivative w.r.t. coordinates."""


class OutputVariableCategory(IntEnum):
    """Defines the category of the output variable."""

    OUT = OutputVariableOperation._NONE
    """Output variable. (e.g. atom energy)"""
    REDU = OutputVariableOperation.REDU
    """Reduced output variable. (e.g. system energy)"""
    DERV_R = OutputVariableOperation.DERV_R
    """Negative derivative w.r.t. coordinates. (e.g. force)"""
    DERV_C = OutputVariableOperation.DERV_C
    """Atomic component of the virial, see PRB 104, 224202 (2021)  """
    DERV_C_REDU = OutputVariableOperation.DERV_C | OutputVariableOperation.REDU
    """Virial, the transposed negative gradient with cell tensor times cell tensor, see eq 40 JCP 159, 054801 (2023). """
    DERV_R_DERV_R = OutputVariableOperation.DERV_R | OutputVariableOperation._SEC_DERV_R
    """Hession matrix, the second derivative w.r.t. coordinates."""


class OutputVariableDef:
    """Defines the shape and other properties of the one output variable.

    It is assume that the fitting network output variables for each
    local atom. This class defines one output variable, including its
    name, shape, reducibility and differentiability.

    Parameters
    ----------
    name
          Name of the output variable. Notice that the xxxx_redu,
          xxxx_derv_c, xxxx_derv_r are reserved names that should
          not be used to define variables.
    shape
          The shape of the variable. e.g. energy should be [1],
          dipole should be [3], polarizabilty should be [3,3].
    reduciable
          If the variable is reduced.
    r_differentiable
          If the variable is differentiated with respect to coordinates
          of atoms. Only reduciable variable are differentiable.
          Negative derivative w.r.t. coordinates will be calcualted. (e.g. force)
    c_differentiable
          If the variable is differentiated with respect to the
          cell tensor (pbc case). Only reduciable variable
          are differentiable.
          Virial, the transposed negative gradient with cell tensor times
          cell tensor, will be calculated, see eq 40 JCP 159, 054801 (2023).
    atomic : bool
          If the variable is defined for each atom.
    category : int
          The category of the output variable.
    hessian : bool
          If hessian is requred
    """

    def __init__(
        self,
        name: str,
        shape: List[int],
        reduciable: bool = False,
        r_differentiable: bool = False,
        c_differentiable: bool = False,
        atomic: bool = True,
        category: int = OutputVariableCategory.OUT.value,
        r_hessian: bool = False,
    ):
        self.name = name
        self.shape = list(shape)
        # jit doesn't support math.prod(self.shape)
        self.output_size = 1
        len_shape = len(self.shape)
        for i in range(len_shape):
            self.output_size *= self.shape[i]
        self.atomic = atomic
        self.reduciable = reduciable
        self.r_differentiable = r_differentiable
        self.c_differentiable = c_differentiable
        if self.c_differentiable and not self.r_differentiable:
            raise ValueError("c differentiable requires r_differentiable")
        if self.reduciable and not self.atomic:
            raise ValueError("a reduciable variable should be atomic")
        self.category = category
        self.r_hessian = r_hessian
        if self.r_hessian:
            if not self.reduciable:
                raise ValueError("only reduciable variable can calculate hessian")
            if not self.r_differentiable:
                raise ValueError("only r_differentiable variable can calculate hessian")


class FittingOutputDef:
    """Defines the shapes and other properties of the fitting network outputs.

    It is assume that the fitting network output variables for each
    local atom. This class defines all the outputs.

    Parameters
    ----------
    var_defs
          List of output variable definitions.

    """

    def __init__(
        self,
        var_defs: List[OutputVariableDef],
    ):
        self.var_defs = {vv.name: vv for vv in var_defs}

    def __getitem__(
        self,
        key: str,
    ) -> OutputVariableDef:
        return self.var_defs[key]

    def get_data(self) -> Dict[str, OutputVariableDef]:
        return self.var_defs

    def keys(self):
        return self.var_defs.keys()


class ModelOutputDef:
    """Defines the shapes and other properties of the model outputs.

    The model reduce and differentiate fitting outputs if applicable.
    If a variable is named by foo, then the reduced variable is called
    foo_redu, the derivative w.r.t. coordinates is called foo_derv_r
    and the derivative w.r.t. cell is called foo_derv_c.

    Parameters
    ----------
    fit_defs
          Definition for the fitting net output

    """

    def __init__(
        self,
        fit_defs: FittingOutputDef,
    ):
        self.def_outp = fit_defs
        self.def_redu = do_reduce(self.def_outp.get_data())
        self.def_derv_r, self.def_derv_c = do_derivative(self.def_outp.get_data())
        self.def_hess_r, _ = do_derivative(self.def_derv_r)
        self.def_derv_c_redu = do_reduce(self.def_derv_c)
        self.var_defs: Dict[str, OutputVariableDef] = {}
        for ii in [
            self.def_outp.get_data(),
            self.def_redu,
            self.def_derv_c,
            self.def_derv_r,
            self.def_derv_c_redu,
            self.def_hess_r,
        ]:
            self.var_defs.update(ii)

    def __getitem__(
        self,
        key: str,
    ) -> OutputVariableDef:
        return self.var_defs[key]

    def get_data(
        self,
        key: str,
    ) -> Dict[str, OutputVariableDef]:
        return self.var_defs

    def keys(self):
        return self.var_defs.keys()

    def keys_outp(self):
        return self.def_outp.keys()

    def keys_redu(self):
        return self.def_redu.keys()

    def keys_derv_r(self):
        return self.def_derv_r.keys()

    def keys_hess_r(self):
        return self.def_hess_r.keys()

    def keys_derv_c(self):
        return self.def_derv_c.keys()

    def keys_derv_c_redu(self):
        return self.def_derv_c_redu.keys()


def get_reduce_name(name: str) -> str:
    return name + "_redu"


def get_deriv_name(name: str) -> Tuple[str, str]:
    return name + "_derv_r", name + "_derv_c"


def get_hessian_name(name: str) -> str:
    return name + "_derv_r_derv_r"


def apply_operation(var_def: OutputVariableDef, op: OutputVariableOperation) -> int:
    """Apply a operation to the category of a variable definition.

    Parameters
    ----------
    var_def : OutputVariableDef
        The variable definition.
    op : OutputVariableOperation
        The operation to be applied.

    Returns
    -------
    int
        The new category of the variable definition.

    Raises
    ------
    ValueError
        If the operation has been applied to the variable definition,
        and exceed the maximum limitation.
    """
    if op == OutputVariableOperation.REDU or op == OutputVariableOperation.DERV_C:
        if check_operation_applied(var_def, op):
            raise ValueError(f"operation {op} has been applied")
    elif op == OutputVariableOperation.DERV_R:
        if check_operation_applied(var_def, OutputVariableOperation.DERV_R):
            op = OutputVariableOperation._SEC_DERV_R
            if check_operation_applied(var_def, OutputVariableOperation._SEC_DERV_R):
                raise ValueError(f"operation {op} has been applied twice")
    else:
        raise ValueError(f"operation {op} not supported")
    return var_def.category | op.value


def check_operation_applied(
    var_def: OutputVariableDef, op: OutputVariableOperation
) -> bool:
    """Check if a operation has been applied to a variable definition.

    Parameters
    ----------
    var_def : OutputVariableDef
        The variable definition.
    op : OutputVariableOperation
        The operation to be checked.

    Returns
    -------
    bool
        True if the operation has been applied, False otherwise.
    """
    return var_def.category & op.value == op.value


def do_reduce(
    def_outp_data: Dict[str, OutputVariableDef],
) -> Dict[str, OutputVariableDef]:
    def_redu: Dict[str, OutputVariableDef] = {}
    for kk, vv in def_outp_data.items():
        if vv.reduciable:
            rk = get_reduce_name(kk)
            def_redu[rk] = OutputVariableDef(
                rk,
                vv.shape,
                reduciable=False,
                r_differentiable=False,
                c_differentiable=False,
                atomic=False,
                category=apply_operation(vv, OutputVariableOperation.REDU),
            )
    return def_redu


def do_derivative(
    def_outp_data: Dict[str, OutputVariableDef],
) -> Tuple[Dict[str, OutputVariableDef], Dict[str, OutputVariableDef]]:
    def_derv_r: Dict[str, OutputVariableDef] = {}
    def_derv_c: Dict[str, OutputVariableDef] = {}
    for kk, vv in def_outp_data.items():
        rkr, rkc = get_deriv_name(kk)
        if vv.r_differentiable:
            def_derv_r[rkr] = OutputVariableDef(
                rkr,
                vv.shape + [3],  # noqa: RUF005
                reduciable=False,
                r_differentiable=(
                    vv.r_hessian and vv.category == OutputVariableCategory.OUT.value
                ),
                c_differentiable=False,
                atomic=True,
                category=apply_operation(vv, OutputVariableOperation.DERV_R),
            )
        if vv.c_differentiable:
            assert vv.r_differentiable
            rkr, rkc = get_deriv_name(kk)
            def_derv_c[rkc] = OutputVariableDef(
                rkc,
                vv.shape + [9],  # noqa: RUF005
                reduciable=True,
                r_differentiable=False,
                c_differentiable=False,
                atomic=True,
                category=apply_operation(vv, OutputVariableOperation.DERV_C),
            )
    return def_derv_r, def_derv_c
