# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
)
from deepmd.dpmodel.utils import (
    nlist_distinguish_types,
)


def make_multi_fitting_model(T_AtomicModel: Type[BaseAtomicModel]):
    class CM(NativeOP, BaseModel):
        def __init__(
            self,
            *args,
            # underscore to prevent conflict with normal inputs
            atomic_model_: Optional[T_AtomicModel] = None,
            **kwargs,
        ):
            BaseModel.__init__(*args, **kwargs)
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISON_DICT
            self.global_np_float_precision = GLOBAL_NP_FLOAT_PRECISION
            self.global_ener_float_precision = GLOBAL_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        def model_output_type(self) -> List[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            vars = [
                kk
                for kk, vv in var_defs.items()
                if vv.category == OutputVariableCategory.OUT
            ]
            return vars

        # def get_out_bias(self) -> torch.Tensor:
        #     return self.atomic_model.get_out_bias()

        # def change_out_bias(
        #     self,
        #     merged,
        #     bias_adjust_mode="change-by-statistic",
        # ) -> None:
        #     """Change the output bias of atomic model according to the input data and the pretrained model.

        #     Parameters
        #     ----------
        #     merged : Union[Callable[[], List[dict]], List[dict]]
        #         - List[dict]: A list of data samples from various data systems.
        #             Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
        #             originating from the `i`-th data system.
        #         - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
        #             only when needed. Since the sampling process can be slow and memory-intensive,
        #             the lazy function helps by only sampling once.
        #     bias_adjust_mode : str
        #         The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
        #         'change-by-statistic' : perform predictions on labels of target dataset,
        #                 and do least square on the errors to obtain the target shift as bias.
        #         'set-by-statistic' : directly use the statistic output bias in the target dataset.
        #     """
        #     self.atomic_model.change_out_bias(
        #         merged,
        #         bias_adjust_mode=bias_adjust_mode,
        #     )

        def input_type_cast(
            self,
            coord: np.ndarray,
            box: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
        ) -> Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = self.reverse_precision_dict[
                self.precision_dict[coord.dtype.name]
            ]
            _lst: List[Optional[np.ndarray]] = [
                vv.astype(coord.dtype) if vv is not None else None
                for vv in [box, fparam, aparam]
            ]
            box, fparam, aparam = _lst
            if (
                input_prec
                == self.reverse_precision_dict[self.global_np_float_precision]
            ):
                return coord, box, fparam, aparam, input_prec
            else:
                pp = self.global_np_float_precision
                return (
                    coord.astype(pp),
                    box.astype(pp) if box is not None else None,
                    fparam.astype(pp) if fparam is not None else None,
                    aparam.astype(pp) if aparam is not None else None,
                    input_prec,
                )

        def output_type_cast(
            self,
            model_ret: Dict[str, np.ndarray],
            input_prec: str,
        ) -> Dict[str, np.ndarray]:
            """Convert the model output to the input prec."""
            do_cast = (
                input_prec
                != self.reverse_precision_dict[self.global_np_float_precision]
            )
            pp = self.precision_dict[input_prec]
            odef = self.model_output_def()
            for kk in odef.keys():
                if kk not in model_ret:
                    # do not return energy_derv_c if not do_atomic_virial
                    continue
                if check_operation_applied(odef[kk], OutputVariableOperation.REDU):
                    model_ret[kk] = (
                        model_ret[kk].to(self.global_ener_float_precision)
                        if model_ret[kk] is not None
                        else None
                    )
                elif do_cast:
                    model_ret[kk] = (
                        model_ret[kk].to(pp) if model_ret[kk] is not None else None
                    )
            return model_ret

        def format_nlist(
            self,
            extended_coord: np.ndarray,
            extended_atype: np.ndarray,
            nlist: np.ndarray,
        ):
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preseved.

            Known limitations:

            In the case of not self.mixed_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel

            Returns
            -------
            formated_nlist
                the formated nlist.

            """
            mixed_types = self.mixed_types()
            ret = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if not mixed_types:
                ret = nlist_distinguish_types(ret, extended_atype, self.get_sel())
            return ret

        def _format_nlist(
            self,
            extended_coord: np.ndarray,
            nlist: np.ndarray,
            nnei: int,
        ):
            n_nf, n_nloc, n_nnei = nlist.shape
            extended_coord = extended_coord.reshape([n_nf, -1, 3])
            rcut = self.get_rcut()

            if n_nnei < nnei:
                # make a copy before revise
                ret = np.concatenate(
                    [
                        nlist,
                        -1 * np.ones([n_nf, n_nloc, nnei - n_nnei], dtype=nlist.dtype),
                    ],
                    axis=-1,
                )
            elif n_nnei > nnei:
                # make a copy before revise
                m_real_nei = nlist >= 0
                ret = np.where(m_real_nei, nlist, 0)
                coord0 = extended_coord[:, :n_nloc, :]
                index = ret.reshape(n_nf, n_nloc * n_nnei, 1).repeat(3, axis=2)
                coord1 = np.take_along_axis(extended_coord, index, axis=1)
                coord1 = coord1.reshape(n_nf, n_nloc, n_nnei, 3)
                rr = np.linalg.norm(coord0[:, :, None, :] - coord1, axis=-1)
                rr = np.where(m_real_nei, rr, float("inf"))
                rr, ret_mapping = np.sort(rr, axis=-1), np.argsort(rr, axis=-1)
                ret = np.take_along_axis(ret, ret_mapping, axis=2)
                ret = np.where(rr > rcut, -1, ret)
                ret = ret[..., :nnei]
            else:  # n_nnei == nnei:
                ret = nlist
            assert ret.shape[-1] == nnei
            return ret

        def do_grad_r(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.
            """
            return self.atomic_model.do_grad_r(var_name)

        def do_grad_c(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.
            """
            return self.atomic_model.do_grad_c(var_name)

        def serialize(self) -> dict:
            return self.atomic_model.serialize()

        @classmethod
        def deserialize(cls, data) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""
            return self.atomic_model.get_dim_aparam()

        def get_sel_type(self) -> List[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """
            return self.atomic_model.get_sel_type()

        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """
            return self.atomic_model.is_aparam_nall()

        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            return self.atomic_model.get_rcut()

        def get_type_map(self) -> List[str]:
            """Get the type map."""
            return self.atomic_model.get_type_map()

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nsel()

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nnei()

        def get_sel(self) -> List[int]:
            """Returns the number of selected atoms for each type."""
            return self.atomic_model.get_sel()

        def mixed_types(self) -> bool:
            """If true, the model
            1. assumes total number of atoms aligned across frames;
            2. uses a neighbor list that does not distinguish different atomic types.

            If false, the model
            1. assumes total number of atoms of each atom type aligned across frames;
            2. uses a neighbor list that distinguishes different atomic types.

            """
            return self.atomic_model.mixed_types()

        def has_message_passing(self) -> bool:
            """Returns whether the model has message passing."""
            return self.atomic_model.has_message_passing()

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model."""
            return self.atomic_model.atomic_output_def()

        def get_ntypes(self) -> int:
            """Get the number of types."""
            return len(self.get_type_map())

        @staticmethod
        def make_pairs(nlist, mapping):
            """Return the pairs from nlist and mapping.

            Returns
            -------
            pairs
                [[i1, j1, 0], [i2, j2, 0], ...],
                in which i and j are the local indices of the atoms

            """
            nframes, nloc, nsel = nlist.shape
            assert nframes == 1
            nlist_reshape = np.reshape(nlist, [nframes, nloc * nsel])
            # nlist is pad with -1
            mask = nlist_reshape >= 0

            ii = np.arange(nloc, dtype=np.int64)
            ii = np.tile(ii.reshape(-1, 1), [1, nsel])
            ii = np.reshape(ii, [nframes, nloc * nsel])
            # nf x (nloc x nsel)
            sel_ii = ii[mask]
            # sel_ii = np.reshape(sel_ii, [nframes, -1, 1])

            # nf x (nloc x nsel)
            sel_nlist = nlist_reshape[mask]
            sel_jj = np.take_along_axis(mapping, sel_nlist, axis=1)
            sel_jj = np.reshape(sel_jj, [nframes, -1])

            # nframes x (nloc x nsel) x 3
            pairs = np.zeros([nframes, nloc * nsel], dtype=np.int64)
            pairs = np.stack((sel_ii, sel_jj, pairs[mask]))

            # select the pair with jj > ii
            # nframes x (nloc x nsel)
            mask = pairs[..., 1] > pairs[..., 0]
            pairs = pairs[mask]
            pairs = np.reshape(pairs, [nframes, -1, 3])

            return pairs

    return CM
