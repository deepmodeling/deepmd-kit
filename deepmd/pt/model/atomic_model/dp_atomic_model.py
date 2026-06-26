# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    Optional,
)

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
        ntypes = len(type_map)
        self.type_map = type_map
        self.ntypes = ntypes
        self.descriptor = descriptor
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.fitting_net = fitting
        if hasattr(self.fitting_net, "reinit_exclude"):
            self.fitting_net.reinit_exclude(self.atom_exclude_types)
        super().init_out_stat()
        self.add_chg_spin_ebd: bool = getattr(
            self.descriptor, "add_chg_spin_ebd", False
        )

    @torch.jit.export
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

    def get_sel(self) -> list[int]:
        """Get the neighbor selection."""
        return self.sel

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

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Optional["DPAtomicModel"] = None,
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.descriptor.change_type_map(
            type_map=type_map,
            model_with_new_type_stat=model_with_new_type_stat.descriptor
            if model_with_new_type_stat is not None
            else None,
        )
        self.fitting_net.change_type_map(type_map=type_map)
        # Reinitialize fitting to get correct sel_type
        if hasattr(self.fitting_net, "reinit_exclude"):
            self.fitting_net.reinit_exclude(self.atom_exclude_types)

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return self.descriptor.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return self.descriptor.need_sorted_nlist_for_lower()

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "standard",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "fitting": self.fitting_net.serialize(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "DPAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        fitting_obj = BaseFitting.deserialize(data.pop("fitting"))
        data["descriptor"] = descriptor_obj
        data["fitting"] = fitting_obj
        obj = super().deserialize(data)
        return obj

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
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
        return_atomic_feature: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coordinates in extended region
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
        return_atomic_feature
            When True, run the fitting net only up to its last hidden layer with
            no force/virial autograd, and additionally return the raw per-atom
            ``descriptor`` and the last hidden ``atomic_feature``. Used by the
            embedding path.

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`. When
            ``return_atomic_feature`` is True, it also contains ``descriptor``
            and ``atomic_feature``.

        """
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        # The embedding path produces no force and never allocates an autograd leaf.
        if (
            not return_atomic_feature
            and (self.do_grad_r() or self.do_grad_c())
            and not extended_coord.requires_grad
        ):
            extended_coord = extended_coord.clone().requires_grad_(True)

        # Handle default chg_spin if descriptor supports it
        if self.add_chg_spin_ebd and charge_spin is None:
            default_cs_tensor = self.descriptor.get_default_chg_spin()
            if default_cs_tensor is not None:
                default_cs_tensor = default_cs_tensor.to(device=extended_coord.device)
                charge_spin = torch.tile(default_cs_tensor.unsqueeze(0), [nframes, 1])

        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
            charge_spin=charge_spin if self.add_chg_spin_ebd else None,
        )
        assert descriptor is not None
        if return_atomic_feature:
            fit_ret = self.fitting_net(
                descriptor,
                atype,
                gr=rot_mat,
                g2=g2,
                h2=h2,
                fparam=fparam,
                aparam=aparam,
                return_atomic_feature=True,
            )
            fit_ret["descriptor"] = descriptor
            return fit_ret
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

    def forward_common_atomic_flat(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_batch: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        extended_ptr: torch.Tensor | None = None,
        central_ext_index: torch.Tensor | None = None,
        nlist_ext: torch.Tensor | None = None,
        a_nlist: torch.Tensor | None = None,
        a_nlist_ext: torch.Tensor | None = None,
        nlist_mask: torch.Tensor | None = None,
        a_nlist_mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        angle_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with flat mixed-nloc batch format."""
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)

        nframes = ptr.numel() - 1
        if self.add_chg_spin_ebd and charge_spin is None:
            default_cs_tensor = self.descriptor.get_default_chg_spin()
            if default_cs_tensor is not None:
                charge_spin = torch.tile(
                    default_cs_tensor.to(device=extended_coord.device).unsqueeze(0),
                    [nframes, 1],
                )

        descriptor_out = self.descriptor.forward_flat(
            extended_coord,
            extended_atype,
            extended_batch,
            nlist,
            mapping,
            batch,
            ptr,
            fparam=fparam,
            charge_spin=charge_spin if self.add_chg_spin_ebd else None,
            central_ext_index=central_ext_index,
            nlist_ext=nlist_ext,
            a_nlist=a_nlist,
            a_nlist_ext=a_nlist_ext,
            nlist_mask=nlist_mask,
            a_nlist_mask=a_nlist_mask,
            edge_index=edge_index,
            angle_index=angle_index,
        )

        descriptor = descriptor_out.get("descriptor")
        rot_mat = descriptor_out.get("rot_mat")
        g2 = descriptor_out.get("g2")
        h2 = descriptor_out.get("h2")

        if central_ext_index is None:
            from deepmd.pt.utils.nlist import (
                get_central_ext_index,
            )

            central_ext_index = get_central_ext_index(extended_batch, ptr)
        atype = extended_atype[central_ext_index]

        fit_ret = self.fitting_net.forward_flat(
            descriptor,
            atype,
            batch,
            ptr,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        fit_ret = self.apply_out_stat(fit_ret, atype)

        atom_mask = self.make_atom_mask(atype).to(torch.int32)
        if self.atom_excl is not None:
            atom_mask *= self.atom_excl(atype.unsqueeze(0)).squeeze(0)

        for kk in fit_ret.keys():
            out_shape = fit_ret[kk].shape
            out_shape2 = 1
            for ss in out_shape[1:]:
                out_shape2 *= ss
            fit_ret[kk] = (
                fit_ret[kk].reshape([out_shape[0], out_shape2]) * atom_mask[:, None]
            ).view(out_shape)
        fit_ret["mask"] = atom_mask

        return fit_ret

    def has_embedding(self) -> bool:
        """A standard descriptor-fitting atomic model supports embeddings."""
        return True

    def forward_embedding(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings, reusing the descriptor and fitting forward.

        The neighbor/type masking mirrors `forward_common_atomic` so that the
        descriptor matches the energy forward, and the heavy descriptor and
        fitting work is delegated to `forward_atomic` with
        ``return_atomic_feature=True``.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall, 3).
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Extended-to-local index map with shape (nf, nall), or None.
        fparam
            Frame parameters with shape (nf, dim_fparam), or None.
        aparam
            Atomic parameters with shape (nf, nloc, dim_aparam), or None.
        charge_spin
            Frame-level charge and spin conditions with shape (nf, 2), or None.

        Returns
        -------
        dict[str, torch.Tensor]
            ``descriptor`` with shape (nf, nloc, d), ``atomic_feature`` (the last
            fitting hidden activation) with shape (nf, nloc, h), and
            ``structural_feature`` (the masked atom-sum of ``atomic_feature``)
            with shape (nf, h).
        """
        _, nloc, _ = nlist.shape
        # Original local types drive the output mask; masked types feed the nets.
        atype = extended_atype[:, :nloc]
        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            nlist = torch.where(pair_mask == 1, nlist, -1)
        ext_atom_mask = self.make_atom_mask(extended_atype)
        fit_ret = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            return_atomic_feature=True,
        )
        atomic_feature = fit_ret["atomic_feature"]
        # nf x nloc
        atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
        if self.atom_excl is not None:
            atom_mask = atom_mask * self.atom_excl(atype)
        structural_feature = (
            atomic_feature * atom_mask[:, :, None].to(atomic_feature.dtype)
        ).sum(dim=1)
        return {
            "descriptor": fit_ret["descriptor"],
            "atomic_feature": atomic_feature,
            "structural_feature": structural_feature,
        }

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict]],
        stat_file_path: DPPath | None = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: list[str] | None = None,
    ) -> None:
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
        compute_or_load_out_stat : bool
            Whether to compute the output statistics.
            If False, it will only compute the input statistics (e.g. mean and standard deviation of descriptors).
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        wrapped_sampler = self._make_wrapped_sampler(sampled_func)
        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.compute_fitting_input_stat(wrapped_sampler, stat_file_path)
        if compute_or_load_out_stat:
            self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

        self._collect_and_set_observed_type(
            wrapped_sampler, stat_file_path, preset_observed_type
        )

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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The dictionary of paths to the statistics files.
        """
        self.fitting_net.compute_input_stats(
            sample_merged,
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.fitting_net.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        """Check if the model has default frame parameters."""
        return self.fitting_net.has_default_fparam()

    def get_default_fparam(self) -> torch.Tensor | None:
        return self.fitting_net.get_default_fparam()

    @torch.jit.export
    def has_chg_spin_ebd(self) -> bool:
        """Check if the model has charge spin embedding."""
        return self.add_chg_spin_ebd

    @torch.jit.export
    def get_dim_chg_spin(self) -> int:
        """Get the dimension of charge_spin input."""
        if self.add_chg_spin_ebd:
            return self.descriptor.get_dim_chg_spin()
        return 0

    @torch.jit.export
    def has_default_chg_spin(self) -> bool:
        """Check if the model has default charge_spin values."""
        if self.add_chg_spin_ebd:
            return self.descriptor.has_default_chg_spin()
        return False

    @torch.jit.export
    def get_default_chg_spin(self) -> torch.Tensor | None:
        """Get the default charge_spin values as a tensor."""
        if self.add_chg_spin_ebd and self.descriptor.has_default_chg_spin():
            return self.descriptor.get_default_chg_spin()
        return None

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.fitting_net.get_dim_aparam()

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
