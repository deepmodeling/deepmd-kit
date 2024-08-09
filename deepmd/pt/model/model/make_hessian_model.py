# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import math
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    get_hessian_name,
)


def compute_hessian(func, inputs):  # anchor created
    # print(f"func in compute_hessian of {type(func)}: {func}")
    # print(f"inputs in compute_hessian of {type(inputs)}: {inputs}")
    device = torch.device('cuda:0')
    inputs.to(device)
    inputs = inputs.requires_grad_(True)
    y = func(inputs)
    grads = torch.autograd.grad(y, inputs, create_graph=True)[0]
    # for j in range(len(inputs)):
    #     grad_j = torch.autograd.grad(y, inputs[j], retain_graph=True, allow_unused=True)[0]
    #     print(f"{j} grad: {grad_j}")
    n = len(inputs)
    hessian = torch.zeros(n, n, device=device)
    for i in range(n):
        grad2 = torch.autograd.grad(grads[i], inputs, retain_graph=True, create_graph=True)[0]
        # print(f"{i} grad2: {grad2}")
        # for j in range(len(inputs)):
        #     grad_ij = torch.autograd.grad(grads[i], inputs[j], retain_graph=True, allow_unused=True)[0]
        #     print(f"{i}{j} grad2: {grad_ij}")
        hessian[i] = grad2
    return hessian


def make_hessian_model(T_Model):
    """Make a model that can compute Hessian.

    LIMITATION: this model is not jitable due to the restrictions of torch jit script.

    LIMITATION: only the hessian of `forward_common` is available.

    Parameters
    ----------
    T_Model
        The model. Should provide the `forward_common` and `atomic_output_def` methods

    Returns
    -------
    The model computes hessian.

    """

    class CM(T_Model):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )
            self.hess_fitting_def = copy.deepcopy(super().atomic_output_def())

        def requires_hessian(
            self,
            keys: Union[str, List[str]],
        ):
            """Set which output variable(s) requires hessian."""
            if isinstance(keys, str):
                keys = [keys]
            for kk in self.hess_fitting_def.keys():
                if kk in keys:
                    self.hess_fitting_def[kk].r_hessian = True

        def atomic_output_def(self):
            """Get the fitting output def."""
            return self.hess_fitting_def

        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            ret = super().forward_common(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )
            vdef = self.atomic_output_def()
            hess_yes = [vdef[kk].r_hessian for kk in vdef.keys()]
            if any(hess_yes):
                hess = self._cal_hessian_all(
                    coord,
                    atype,
                    box=box,
                    fparam=fparam,
                    aparam=aparam,
                )
                ret.update(hess)
            return ret

        def _cal_hessian_all(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            nf, nloc = atype.shape
            coord = coord.view([nf, (nloc * 3)])
            box = box.view([nf, 9]) if box is not None else None
            fparam = fparam.view([nf, -1]) if fparam is not None else None
            aparam = aparam.view([nf, nloc, -1]) if aparam is not None else None
            fdef = self.atomic_output_def()
            # print(f"fdef in _cal_hessian_all: {fdef}")  # anchor added
            # print(f"keys of fdef in _cal_hessian_all: {fdef.keys()}")  # anchor added ['energy', 'mask']
            # keys of values that require hessian
            hess_keys: List[str] = []
            for kk in fdef.keys():
                if fdef[kk].r_hessian:
                    hess_keys.append(kk)
            # result dict init by empty lists
            res = {get_hessian_name(kk): [] for kk in hess_keys}
            # loop over variable
            for kk in hess_keys:
                vdef = fdef[kk]
                vshape = vdef.shape
                vsize = math.prod(vdef.shape)
                # print(f"vdef is {vdef}, vshape is {vshape}, vsize is {vsize}")  # anchor added
                # loop over frames
                for ii in range(nf):
                    icoord = coord[ii]
                    iatype = atype[ii]
                    ibox = box[ii] if box is not None else None
                    ifparam = fparam[ii] if fparam is not None else None
                    iaparam = aparam[ii] if aparam is not None else None
                    # loop over all components
                    for idx in range(vsize):
                        hess = self._cal_hessian_one_component(
                            idx, icoord, iatype, ibox, ifparam, iaparam
                        )
                        res[get_hessian_name(kk)].append(hess)
                res[get_hessian_name(kk)] = torch.stack(res[get_hessian_name(kk)]).view(
                    (nf, *vshape, nloc * 3, nloc * 3)
                )
            return res

        def _cal_hessian_one_component(
            self,
            ci,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # coord, # (nloc x 3)
            # atype, # nloc
            # box: Optional[torch.Tensor] = None,     # 9
            # fparam: Optional[torch.Tensor] = None,  # nfp
            # aparam: Optional[torch.Tensor] = None,  # (nloc x nap)
            # print(f"coord in _cal_hessian_one_component: {coord}")  # anchor added
            # print(f"atype in _cal_hessian_one_component: {atype}")  # anchor added
            # print(f"box in _cal_hessian_one_component: {box}")  # anchor added
            wc = wrapper_class_forward_energy(self, ci, atype, box, fparam, aparam)

            # def energy_func(x):  # anchor trying: success
            #     return (x ** 2).sum()
            # def energy_func(x):  # anchor trying: success
            #     return (2*x.pow(2)+3*x.pow(2)+x.pow(3)).sum()
            # def energy_func(x):  # anchor trying: success
            #     return ((x[0].pow(3))+5*(x[1].pow(2))+2*(x[0].pow(2))*(x[2].pow(5)))
            # wc = energy_func
            # print(f"wrapper_class in _cal_hessian_one_component: {type(wc)}")  # anchor added
            # print(f"wc in _cal_hessian_one_component: {wc}")  # anchor added

            # hess = torch.autograd.functional.hessian(
            #     wc,
            #     coord,
            #     create_graph=False,
            #     # create_graph=True,  # anchor changed to: FloatingPointError: gradients are Nan/Inf
            # )
            # jacobian = torch.autograd.functional.jacobian(  # anchor trying
            #     wc,
            #     coord,
            #     create_graph=True,
            # )
            # print(f"jacobian in _cal_hessian_one_component: {jacobian}")  # anchor added
            # jacobian.backward(torch.ones_like(coord))
            # tmp = torch.autograd.grad(jacobian[0], coord[0], create_graph=False, allow_unused=True)  # anchor added
            # er_from_wc = wc.__call__(coord)  # anchor added
            # print(f"er_from_wc in _cal_hessian_one_component: {er_from_wc}")  # anchor added
            # coord = torch.tensor(coord, requires_grad=True)  # anchor trying
            # tmp = torch.autograd.grad(er_from_wc, coord[3], create_graph=False, allow_unused=True)  # anchor added
            # print(f"tmp in _cal_hessian_one_component: {tmp}")  # anchor added
            # coord = torch.tensor(coord, requires_grad=True)  # anchor trying
            # print(f"len of coord is {len(coord)}")  # anchor
            hess = compute_hessian(wc, coord)  # anchor trying: identical to t.ag.f.hessian
            # hess = torch.func.hessian(wc)(coord)  # anchor tried
            # print(f"hessian in _cal_hessian_one_component: {hess}")  # anchor added
            return hess

    class wrapper_class_forward_energy:
        # torch.autograd.set_detect_anomaly(True)  # anchor added
        def __init__(
            self,
            obj: CM,
            ci: int,
            atype: torch.Tensor,
            box: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor],
            aparam: Optional[torch.Tensor],
        ):
            self.atype, self.box, self.fparam, self.aparam = atype, box, fparam, aparam
            self.ci = ci
            self.obj = obj

        def __call__(
            self,
            xx,
        ):
            ci = self.ci
            atype, box, fparam, aparam = self.atype, self.box, self.fparam, self.aparam
            res = super(CM, self.obj).forward_common(
                xx.unsqueeze(0),
                atype.unsqueeze(0),
                box.unsqueeze(0) if box is not None else None,
                fparam.unsqueeze(0) if fparam is not None else None,
                aparam.unsqueeze(0) if aparam is not None else None,
                do_atomic_virial=False,
            )
            er = res["energy_redu"][0].view([-1])[ci]
            # def energy_func(x):  # anchor trying: success
            #     return (x ** 2).sum()
            # res = energy_func(xx)  # anchor added
            # er = res  # anchor added
            # print(f"obj in wrapper_class_forward_energy: {self.obj}")  # anchor added
            # print(f"res in wrapper_class_forward_energy: {res}")  # anchor added
            # print(f"er in wrapper_class_forward_energy: {er}")  # anchor added
            # print(f"er grad_fn in wrapper_class_forward_energy: {er.grad_fn}")  # anchor added
            return er

    return CM

