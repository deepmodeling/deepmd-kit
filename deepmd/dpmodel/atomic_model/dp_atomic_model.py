# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
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


@BaseAtomicModel.register("standard")
class DPAtomicModel(BaseAtomicModel):
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
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(type_map, **kwargs)
        self.descriptor = descriptor
        self.fitting_net = fitting
        if hasattr(self.fitting_net, "reinit_exclude"):
            self.fitting_net.reinit_exclude(self.atom_exclude_types)
        self.type_map = type_map
        super().init_out_stat()

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return self.fitting_net.output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.descriptor.get_rcut()

    def get_sel(self) -> list[int]:
        """Get the neighbor selection."""
        return self.descriptor.get_sel()

    def set_case_embd(self, case_idx: int) -> None:
        """
        Set the case embedding of this atomic model by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        self.fitting_net.set_case_embd(case_idx)

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.descriptor.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return self.descriptor.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return self.descriptor.need_sorted_nlist_for_lower()

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
        self.descriptor.enable_compression(
            min_nbor_dist,
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        )

    def forward_atomic(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
    ) -> dict[str, Array]:
        """Models' atomic predictions.

        Parameters
        ----------
        extended_coord
            coordinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices. nf x nall
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
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
        )
        ret = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        return ret

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None = None,
        compute_or_load_out_stat: bool = True,
    ) -> None:
        """Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The path to the stat file.
        compute_or_load_out_stat : bool
            Whether to compute the output statistics.
            If False, it will only compute the input statistics
            (e.g. mean and standard deviation of descriptors).
        """
        if stat_file_path is not None and self.type_map is not None:
            stat_file_path /= " ".join(self.type_map)

        wrapped_sampler = self._make_wrapped_sampler(sampled_func)
        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.fitting_net.compute_input_stats(
            wrapped_sampler, stat_file_path=stat_file_path
        )
        if compute_or_load_out_stat:
            self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        self.type_map = type_map
        self.descriptor.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat.descriptor
            if model_with_new_type_stat is not None
            else None,
        )
        self.fitting_net.change_type_map(type_map=type_map)

    def compute_fitting_input_stat(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Compute the input statistics (e.g. mean and stddev) for the fittings from packed data.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, ``merged[i]``, is a data dictionary containing
                ``keys``: ``np.ndarray`` originating from the ``i``-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples
                in the above format only when needed.
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        self.fitting_net.compute_input_stats(
            sample_merged,
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )

    def serialize(self) -> dict:
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "type": "standard",
                "@version": 2,
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting_net.serialize(),
            }
        )
        return dd

    # for subclass overridden
    base_descriptor_cls = BaseDescriptor
    """The base descriptor class."""
    base_fitting_cls = BaseFitting
    """The base fitting class."""

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DPAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 2)
        data.pop("@class")
        data.pop("type")
        descriptor_obj = cls.base_descriptor_cls.deserialize(data.pop("descriptor"))
        fitting_obj = cls.base_fitting_cls.deserialize(data.pop("fitting"))
        data["descriptor"] = descriptor_obj
        data["fitting"] = fitting_obj
        obj = super().deserialize(data)
        return obj

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return self.fitting_net.has_default_fparam()

    def get_default_fparam(self) -> list[float] | None:
        """Get the default frame parameters."""
        return self.fitting_net.get_default_fparam()

    def get_sel_type(self) -> list[int]:
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
