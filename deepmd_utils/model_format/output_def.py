# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)


def check_var(var, var_def):
    if var_def.atomic:
        # var.shape == [nf, nloc, *var_def.shape]
        if len(var.shape) != len(var_def.shape) + 2:
            raise ValueError(f"{var.shape[2:]} length not matching def {var_def.shape}")
        if list(var.shape[2:]) != var_def.shape:
            raise ValueError(f"{var.shape[2:]} not matching def {var_def.shape}")
    else:
        # var.shape == [nf, *var_def.shape]
        if len(var.shape) != len(var_def.shape) + 1:
            raise ValueError(f"{var.shape[1:]} length not matching def {var_def.shape}")
        if list(var.shape[1:]) != var_def.shape:
            raise ValueError(f"{var.shape[1:]} not matching def {var_def.shape}")


def model_check_output(cls):
    """Check if the output of the Model is consistent with the definition.

    Two methods are assumed to be provided by the Model:
    1. Model.output_def that gives the output definition.
    2. Model.forward that defines the forward path of the model.

    """

    class wrapper(cls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.md = cls.output_def(self)

        def forward(
            self,
            *args,
            **kwargs,
        ):
            ret = cls.forward(self, *args, **kwargs)
            for kk in self.md.keys_outp():
                dd = self.md[kk]
                check_var(ret[kk], dd)
                if dd.reduciable:
                    rk = get_reduce_name(kk)
                    check_var(ret[rk], self.md[rk])
                if dd.differentiable:
                    dnr, dnc = get_deriv_name(kk)
                    check_var(ret[dnr], self.md[dnr])
                    check_var(ret[dnc], self.md[dnc])
            return ret

    return wrapper


def fitting_check_output(cls):
    """Check if the output of the Fitting is consistent with the definition.

    Two methods are assumed to be provided by the Fitting:
    1. Fitting.output_def that gives the output definition.
    2. Fitting.forward defines the forward path of the fitting.

    """

    class wrapper(cls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.md = cls.output_def(self)

        def forward(
            self,
            *args,
            **kwargs,
        ):
            ret = cls.forward(self, *args, **kwargs)
            for kk in self.md.keys():
                dd = self.md[kk]
                check_var(ret[kk], dd)
            return ret

    return wrapper


class VariableDef:
    """Defines the shape and other properties of a variable.

    Parameters
    ----------
    name
          Name of the output variable. Notice that the xxxx_redu,
          xxxx_derv_c, xxxx_derv_r are reserved names that should
          not be used to define variables.
    shape
          The shape of the variable. e.g. energy should be [1],
          dipole should be [3], polarizabilty should be [3,3].
    atomic
          If the variable is defined for each atom.

    """

    def __init__(
        self,
        name: str,
        shape: Union[List[int], Tuple[int]],
        atomic: bool = True,
    ):
        self.name = name
        self.shape = list(shape)
        self.atomic = atomic


class OutputVariableDef(VariableDef):
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
    differentiable
          If the variable is differentiated with respect to coordinates
          of atoms and cell tensor (pbc case). Only reduciable variable
          are differentiable.

    """

    def __init__(
        self,
        name: str,
        shape: Union[List[int], Tuple[int]],
        reduciable: bool = False,
        differentiable: bool = False,
    ):
        # fitting output must be atomic
        super().__init__(name, shape, atomic=True)
        self.reduciable = reduciable
        self.differentiable = differentiable
        if not self.reduciable and self.differentiable:
            raise ValueError("only reduciable variable are differentiable")


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
        var_defs: List[OutputVariableDef] = [],
    ):
        self.var_defs = {vv.name: vv for vv in var_defs}

    def __getitem__(
        self,
        key,
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
        self.def_redu = do_reduce(self.def_outp)
        self.def_derv_r, self.def_derv_c = do_derivative(self.def_outp)
        self.var_defs = {}
        for ii in [
            self.def_outp.get_data(),
            self.def_redu,
            self.def_derv_c,
            self.def_derv_r,
        ]:
            self.var_defs.update(ii)

    def __getitem__(self, key) -> VariableDef:
        return self.var_defs[key]

    def get_data(self, key) -> Dict[str, VariableDef]:
        return self.var_defs

    def keys(self):
        return self.var_defs.keys()

    def keys_outp(self):
        return self.def_outp.keys()

    def keys_redu(self):
        return self.def_redu.keys()

    def keys_derv_r(self):
        return self.def_derv_r.keys()

    def keys_derv_c(self):
        return self.def_derv_c.keys()


def get_reduce_name(name):
    return name + "_redu"


def get_deriv_name(name):
    return name + "_derv_r", name + "_derv_c"


def do_reduce(
    def_outp,
):
    def_redu = {}
    for kk, vv in def_outp.get_data().items():
        if vv.reduciable:
            rk = get_reduce_name(kk)
            def_redu[rk] = VariableDef(rk, vv.shape, atomic=False)
    return def_redu


def do_derivative(
    def_outp,
):
    def_derv_r = {}
    def_derv_c = {}
    for kk, vv in def_outp.get_data().items():
        if vv.differentiable:
            rkr, rkc = get_deriv_name(kk)
            def_derv_r[rkr] = VariableDef(rkr, [*vv.shape, 3], atomic=True)
            def_derv_c[rkc] = VariableDef(rkc, [*vv.shape, 3, 3], atomic=False)
    return def_derv_r, def_derv_c
