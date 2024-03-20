# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import functools
import logging
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)

log = logging.getLogger(__name__)


@BaseAtomicModel.register("standard")
class DPAtomicModel(torch.nn.Module, BaseAtomicModel):
    """Model give atomic prediction of some physical property.

    Parameters
    ----------
    descriptor
            Descriptor
    fitting_net
            Fitting net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    """

    def __init__(
        self,
        descriptor,
        fitting,
        type_map: List[str],
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        self.descriptor = descriptor
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.fitting_net = fitting
        # order matters ntypes and type_map should be initialized first.
        BaseAtomicModel.__init__(self, **kwargs)

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return (
            self.fitting_net.output_def()
            if self.fitting_net is not None
            else self.coord_denoise_net.output_def()
        )

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    @torch.jit.export
    def get_type_map(self) -> List[str]:
        """Get the type map."""
        return self.type_map

    def get_sel(self) -> List[int]:
        """Get the neighbor selection."""
        return self.sel

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.descriptor.mixed_types()

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 1,
                "type": "standard",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting_net.serialize(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        fitting_obj = BaseFitting.deserialize(data.pop("fitting"))
        type_map = data.pop("type_map", None)
        obj = cls(descriptor_obj, fitting_obj, type_map=type_map, **data)
        return obj

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        assert descriptor is not None
        # energy, force
        fit_ret = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        return fit_ret

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        @functools.lru_cache
        def wrapped_sampler():
            sampled = sampled_func()
            if self.pair_excl is not None:
                pair_exclude_types = self.pair_excl.get_exclude_types()
                for sample in sampled:
                    sample["pair_exclude_types"] = list(pair_exclude_types)
            if self.atom_excl is not None:
                atom_exclude_types = self.atom_excl.get_exclude_types()
                for sample in sampled:
                    sample["atom_exclude_types"] = list(atom_exclude_types)
            return sampled

        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        if self.fitting_net is not None:
            self.fitting_net.compute_output_stats(wrapped_sampler, stat_file_path)

    def change_out_bias(
        self, merged, origin_type_map, full_type_map, bias_shift="delta"
    ) -> None:
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        origin_type_map : List[str]
            The original type_map in dataset, they are targets to change the energy bias.
        full_type_map : List[str]
            The full type_map in pre-trained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        sorter = np.argsort(full_type_map)
        missing_types = [t for t in origin_type_map if t not in full_type_map]
        assert (
            not missing_types
        ), f"Some types are not in the pre-trained model: {list(missing_types)} !"
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        original_bias = self.fitting_net["bias_atom_e"]
        if bias_shift == "delta":

            def model_forward(coord, atype, box, fparam=None, aparam=None):
                with torch.no_grad():  # it's essential for pure torch forward function to use auto_batchsize
                    (
                        extended_coord,
                        extended_atype,
                        mapping,
                        nlist,
                    ) = extend_input_and_build_neighbor_list(
                        coord,
                        atype,
                        self.get_rcut(),
                        self.get_sel(),
                        mixed_types=self.mixed_types(),
                        box=box,
                    )
                    atomic_ret = self.forward_common_atomic(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping=mapping,
                        fparam=fparam,
                        aparam=aparam,
                    )
                    return atomic_ret["energy"].detach()

            delta_bias_e = compute_output_stats(
                merged,
                self.get_ntypes(),
                model_forward=model_forward,
            )
            bias_atom_e = delta_bias_e + original_bias
        elif bias_shift == "statistic":
            bias_atom_e = compute_output_stats(
                merged,
                self.get_ntypes(),
            )
        else:
            raise RuntimeError("Unknown bias_shift mode: " + bias_shift)
        log.info(
            f"Change energy bias of {origin_type_map!s} "
            f"from {to_numpy_array(original_bias[idx_type_map]).reshape(-1)!s} "
            f"to {to_numpy_array(bias_atom_e[idx_type_map]).reshape(-1)!s}."
        )
        self.fitting_net["bias_atom_e"] = bias_atom_e

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.fitting_net.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
