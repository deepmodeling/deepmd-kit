# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.dpmodel.loss.reduction import (
    masked_atom_mean,
    masked_pair_mean,
    per_frame_component_mean,
)
from deepmd.dpmodel.utils.neighbor_graph.graph import (
    frame_id_from_n_node,
)
from deepmd.dpmodel.utils.neighbor_graph.segment import (
    segment_sum,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.loss import (
    resolve_huber_deltas,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def _huber_from_residual(residual: Array, delta: float = 1.0) -> Array:
    xp = array_api_compat.array_namespace(residual)
    abs_error = xp.abs(residual)
    quadratic_loss = 0.5 * residual**2
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = xp.where(abs_error <= delta, quadratic_loss, linear_loss)
    return xp.mean(loss)


def custom_huber_loss(predictions: Array, targets: Array, delta: float = 1.0) -> Array:
    r"""Return the mean Huber loss.

    For residual :math:`e=y-\hat y`, the elementwise loss is

    .. math::

       H_\delta(e)=\begin{cases}
       \tfrac12 e^2,& |e|\le\delta,\\
       \delta(|e|-\tfrac12\delta),& |e|>\delta.
       \end{cases}
    """
    return _huber_from_residual(targets - predictions, delta)


class EnergyLoss(Loss):
    r"""Construct a layer to compute loss on energy, force and virial.

    The total objective is a weighted sum of the enabled error terms,

    .. math::

       L=p_E L_E+p_F L_F+p_\Xi L_\Xi+p_{E_i}L_{E_i}
       +p_{PF}L_{PF}+p_{GF}L_{GF}.

    Each prefactor is interpolated using the current learning rate
    :math:`\eta` as

    .. math::

       p(\eta)=p_{\mathrm{limit}}+
       (p_{\mathrm{start}}-p_{\mathrm{limit}})
       \frac{\eta}{\eta_0}.

    The individual terms are mean squared, mean absolute, or Huber errors as
    configured.  In relative-force mode, each force residual is divided by
    :math:`\lVert\hat{\mathbf F}_i\rVert+\nu`, where :math:`\nu` is
    ``relative_f``.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    start_pref_e : float
        The prefactor of energy loss at the start of the training.
    limit_pref_e : float
        The prefactor of energy loss at the end of the training.
    start_pref_f : float
        The prefactor of force loss at the start of the training.
    limit_pref_f : float
        The prefactor of force loss at the end of the training.
    start_pref_v : float
        The prefactor of virial loss at the start of the training.
    limit_pref_v : float
        The prefactor of virial loss at the end of the training.
    start_pref_ae : float
        The prefactor of atomic energy loss at the start of the training.
    limit_pref_ae : float
        The prefactor of atomic energy loss at the end of the training.
    start_pref_pf : float
        The prefactor of atomic prefactor force loss at the start of the training.
    limit_pref_pf : float
        The prefactor of atomic prefactor force loss at the end of the training.
    relative_f : float
        If provided, relative force error will be used in the loss. The difference
        of force will be normalized by the magnitude of the force in the label with
        a shift given by relative_f
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \sum_i c_i E_i
    start_pref_gf : float
        The prefactor of generalized force loss at the start of the training.
    limit_pref_gf : float
        The prefactor of generalized force loss at the end of the training.
    numb_generalized_coord : int
        The dimension of generalized coordinates.
    start_pref_h : float
        The prefactor of Hessian loss at the start of the training.
    limit_pref_h : float
        The prefactor of Hessian loss at the end of the training.
    use_default_pf : bool
        If true, use default atom_pref of 1.0 for all atoms when atom_pref data is not provided.
        This allows using the prefactor force loss (pf) without requiring atom_pref.npy files.
    use_huber : bool
        Enables Huber loss calculation for energy/force/virial terms with user-defined threshold delta (D).
        The loss function smoothly transitions between L2 and L1 loss:
        - For absolute prediction errors within D: quadratic loss (0.5 * (error**2))
        - For absolute errors exceeding D: linear loss (D * |error| - 0.5 * D)
        Formula: loss = 0.5 * (error**2) if |error| <= D else D * (|error| - 0.5 * D).
    huber_delta : float | list[float]
        The threshold delta (D) used for Huber loss, controlling transition between
        L2 and L1 loss. It can be either one float shared by all terms or a list of
        three values ordered as [energy, force, virial].
    loss_func : str
        Loss function type for energy, force, and virial terms.
        Options: 'mse' (Mean Squared Error, L2 loss, default) or 'mae' (Mean Absolute Error, L1 loss).
        MAE loss is less sensitive to outliers compared to MSE loss.
        Future extensions may support additional loss types.
    f_use_norm : bool
        If true, use L2 norm of force vectors for loss calculation when loss_func='mae' or use_huber is True.
        Instead of computing loss on force components, computes loss on ||F_pred - F_label||_2.
        This treats the force vector as a whole rather than three independent components.
    intensive_ener_virial : bool
        If true, the non-Huber MSE energy and virial losses use intensive normalization,
        i.e. a 1/N^2 factor instead of the legacy 1/N scaling. This matches per-atom
        RMSE-style normalization for those terms. MAE and Huber modes use different
        scaling and are not affected in the same way by this flag.
        If false (default), the legacy normalization is used for the affected terms.
        The default is false for backward compatibility with models trained using
        deepmd-kit <= 3.1.3.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.02,
        limit_pref_e: float = 1.00,
        start_pref_f: float = 1000,
        limit_pref_f: float = 1.00,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: float | None = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        start_pref_h: float = 0.0,
        limit_pref_h: float = 0.0,
        use_huber: bool = False,
        huber_delta: float | list[float] = 0.01,
        loss_func: str = "mse",
        f_use_norm: bool = False,
        use_default_pf: bool = False,
        intensive_ener_virial: bool = False,
        **kwargs: Any,
    ) -> None:
        # Validate loss_func
        valid_loss_funcs = ["mse", "mae"]
        if loss_func not in valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func '{loss_func}'. Must be one of {valid_loss_funcs}."
            )

        self.loss_func = loss_func
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.numb_generalized_coord = numb_generalized_coord
        self.start_pref_h = start_pref_h
        self.limit_pref_h = limit_pref_h
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_f = self.start_pref_f != 0.0 or self.limit_pref_f != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0
        self.has_pf = self.start_pref_pf != 0.0 or self.limit_pref_pf != 0.0
        self.has_gf = self.start_pref_gf != 0.0 or self.limit_pref_gf != 0.0
        self.has_h = self.start_pref_h != 0.0 or self.limit_pref_h != 0.0
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.f_use_norm = f_use_norm
        self.use_default_pf = use_default_pf
        self.intensive_ener_virial = intensive_ener_virial
        if self.f_use_norm and not (self.use_huber or self.loss_func == "mae"):
            raise RuntimeError(
                "f_use_norm can only be True when use_huber or loss_func='mae'."
            )
        (
            self._huber_delta_energy,
            self._huber_delta_force,
            self._huber_delta_virial,
        ) = resolve_huber_deltas(huber_delta)
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force."
            )
        if self.use_huber and self.has_h:
            raise RuntimeError("Huber loss is not implemented for hessian.")

    @property
    def supports_ragged_batches(self) -> bool:
        """Whether the configured terms accept a flat per-node batch axis."""
        return not (self.has_gf or self.has_h)

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        r"""Calculate the weighted energy-model objective.

        This evaluates the objective and learning-rate-dependent prefactors
        defined in :class:`EnergyLoss`.  The diagnostics contain per-term RMSE
        values in MSE/Huber mode and per-term MAE values when ``loss_func`` is
        ``"mae"`` or ``mae=True``.  RMSE diagnostics remain ordinary residual
        RMSEs when the optimized objective uses Huber loss.  The aggregate
        ``rmse`` entry is :math:`\sqrt{L}` for the fully weighted objective,
        including all enabled prefactors and any configured Huber terms.
        """
        energy = model_dict["energy"]
        xp = array_api_compat.array_namespace(energy)

        force_required = (
            self.has_f or self.has_pf or self.relative_f is not None or self.has_gf
        )
        if self.has_e:
            energy_hat = label_dict["energy"]
            find_energy = label_dict["find_energy"]
        if force_required:
            force = model_dict["force"]
            force_hat = label_dict["force"]
            find_force = label_dict["find_force"]
        if self.has_v:
            virial = model_dict["virial"]
            virial_hat = label_dict["virial"]
            find_virial = label_dict["find_virial"]
        if self.has_ae:
            atom_ener = model_dict["atom_energy"]
            atom_ener_hat = label_dict["atom_ener"]
            find_atom_ener = label_dict["find_atom_ener"]
        if self.has_pf:
            atom_pref = label_dict["atom_pref"]
            find_atom_pref = (
                label_dict["find_atom_pref"] if not self.use_default_pf else 1.0
            )

        # Two things about a batch decide how its terms reduce, and the node
        # axis states them differently.
        #
        # ``inv``, the reciprocal included-atom count of each frame, is what
        # the extensive frame-level terms (energy, virial) divide by.
        # ``maskf`` marks both padded rows and model-excluded atom types for
        # the per-atom terms. A rectangular batch carries a two-dimensional
        # mask whose row sums give the counts. A ragged batch carries the same
        # information on its flat node axis, with ``n_node`` defining the
        # frame segments.
        maskf = (
            xp.astype(model_dict["mask"], energy.dtype)
            if "mask" in model_dict
            else None
        )
        is_ragged = "n_node" in model_dict
        frame_id = None
        included_n_node = None
        inv = None
        if is_ragged:
            n_node = model_dict["n_node"]
            _nf = n_node.shape[0]
            if maskf is None:
                included_n_node = xp.astype(n_node, energy.dtype)
            else:
                frame_id = frame_id_from_n_node(n_node, n_total=maskf.shape[0])
                included_n_node = segment_sum(xp.reshape(maskf, (-1,)), frame_id, _nf)
        elif maskf is not None:
            included_n_node = xp.reshape(xp.sum(maskf, axis=-1), (-1,))
            _nloc = maskf.shape[1]
        if included_n_node is not None:
            has_included_node = included_n_node > 0
            safe_included_n_node = xp.where(
                has_included_node,
                included_n_node,
                xp.ones_like(included_n_node),
            )
            inv = xp.where(
                has_included_node,
                1.0 / safe_included_n_node,
                xp.zeros_like(included_n_node),
            )
        if maskf is not None:
            _node_shape = maskf.shape
        if inv is not None:
            _nf = inv.shape[0]

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener = model_dict["atom_energy"]
            atom_ener_coeff = label_dict["atom_ener_coeff"]
            atom_ener_coeff = xp.reshape(atom_ener_coeff, atom_ener.shape)
            weighted_atom_ener = atom_ener_coeff * atom_ener
            if is_ragged:
                if frame_id is None:
                    frame_id = frame_id_from_n_node(n_node, n_total=atom_ener.shape[0])
                energy = segment_sum(weighted_atom_ener, frame_id, _nf)
            else:
                energy = xp.sum(weighted_atom_ener, axis=1)
        if force_required:
            force_reshape = xp.reshape(force, (-1,))
            force_hat_reshape = xp.reshape(force_hat, (-1,))
            diff_f = force_hat_reshape - force_reshape
        else:
            diff_f = None

        if self.relative_f is not None:
            force_hat_3 = xp.reshape(force_hat, (-1, 3))
            norm_f = (
                xp.reshape(xp.linalg.vector_norm(force_hat_3, axis=1), (-1, 1))
                + self.relative_f
            )
            diff_f_3 = xp.reshape(diff_f, (-1, 3))
            diff_f_3 = diff_f_3 / norm_f
            diff_f = xp.reshape(diff_f_3, (-1,))

        atom_norm = 1.0 / natoms
        atom_norm_ener = 1.0 / natoms
        lr_ratio = learning_rate / self.starter_learning_rate
        if self.has_e:
            pref_e = find_energy * (
                self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * lr_ratio
            )
        if self.has_f:
            pref_f = find_force * (
                self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * lr_ratio
            )
        if self.has_v:
            pref_v = find_virial * (
                self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * lr_ratio
            )
        if self.has_ae:
            pref_ae = find_atom_ener * (
                self.limit_pref_ae
                + (self.start_pref_ae - self.limit_pref_ae) * lr_ratio
            )
        if self.has_pf:
            effective_find_pf = find_force * find_atom_pref
            pref_pf = effective_find_pf * (
                self.limit_pref_pf
                + (self.start_pref_pf - self.limit_pref_pf) * lr_ratio
            )
        if self.has_h:
            pref_h = (
                self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * lr_ratio
            )

        loss = 0
        more_loss = {}
        # Normalization exponent controls loss scaling with system size:
        # - norm_exp=2 (intensive_ener_virial=True): loss uses 1/N² scaling, making it independent of system size
        # - norm_exp=1 (intensive_ener_virial=False, legacy): loss uses 1/N scaling, which varies with system size
        norm_exp = 2 if self.intensive_ener_virial else 1
        if self.has_e:
            if self.loss_func == "mse":
                l2_ener_loss = xp.mean(xp.square(energy - energy_hat))
                if inv is not None:
                    # Idiom 2 (extensive): per-frame normalization by real-atom count.
                    se = xp.square(energy - energy_hat)  # [nf, k]
                    per_frame = per_frame_component_mean(se)  # [nf]
                    if not self.use_huber:
                        loss += pref_e * xp.mean(per_frame * inv**norm_exp)
                    else:
                        inv_col = xp.reshape(inv, (_nf, 1))  # [nf, 1]
                        l_huber_loss = custom_huber_loss(
                            inv_col * energy,
                            inv_col * energy_hat,
                            delta=self._huber_delta_energy,
                        )
                        loss += pref_e * l_huber_loss
                    more_loss["rmse_e"] = self.display_if_exist(
                        xp.sqrt(xp.mean(per_frame * inv**2)), find_energy
                    )
                else:
                    if not self.use_huber:
                        loss += atom_norm_ener**norm_exp * (pref_e * l2_ener_loss)
                    else:
                        l_huber_loss = custom_huber_loss(
                            atom_norm_ener * energy,
                            atom_norm_ener * energy_hat,
                            delta=self._huber_delta_energy,
                        )
                        loss += pref_e * l_huber_loss
                    more_loss["rmse_e"] = self.display_if_exist(
                        xp.sqrt(l2_ener_loss) * atom_norm_ener, find_energy
                    )
            elif self.loss_func == "mae":
                l1_ener_loss = xp.mean(xp.abs(energy - energy_hat))
                if inv is not None:
                    abs_e = xp.abs(energy - energy_hat)  # [nf, k]
                    per_frame_ae = per_frame_component_mean(abs_e)  # [nf]
                    l1_ener_masked = xp.mean(per_frame_ae * inv)
                    loss += pref_e * l1_ener_masked
                    more_loss["mae_e"] = self.display_if_exist(
                        l1_ener_masked, find_energy
                    )
                else:
                    loss += atom_norm_ener * (pref_e * l1_ener_loss)
                    more_loss["mae_e"] = self.display_if_exist(
                        l1_ener_loss * atom_norm_ener, find_energy
                    )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for energy loss."
                )
            if mae:
                if inv is not None:
                    per_frame_ae = per_frame_component_mean(xp.abs(energy - energy_hat))
                    mae_e = xp.mean(per_frame_ae * inv)
                else:
                    mae_e = xp.mean(xp.abs(energy - energy_hat)) * atom_norm_ener
                more_loss["mae_e"] = self.display_if_exist(mae_e, find_energy)
                mae_e_all = xp.mean(xp.abs(energy - energy_hat))
                more_loss["mae_e_all"] = self.display_if_exist(mae_e_all, find_energy)
        if self.has_f:
            if self.loss_func == "mse":
                l2_force_loss = xp.mean(xp.square(diff_f))
                if maskf is not None:
                    # Idiom 1 (per-atom masked mean, ncomp=3).
                    diff_f_3d = xp.reshape(diff_f, (*_node_shape, 3))
                    # Masked MSE computed for rmse_f display regardless of use_huber.
                    l2_force_masked = masked_atom_mean(xp.square(diff_f_3d), maskf, 3)
                    if not self.use_huber:
                        loss += pref_f * l2_force_masked
                    else:
                        # ``f_use_norm`` selects the residual an atom
                        # contributes: three independent components, or the
                        # single L2 norm of its force-error vector. That choice
                        # sets the label count per atom, which is exactly the
                        # ``ncomp`` the pooled reduction divides by.
                        if not self.f_use_norm:
                            abs_e = xp.abs(diff_f_3d)
                            quad = 0.5 * xp.square(diff_f_3d)
                            lin = self._huber_delta_force * (
                                abs_e - 0.5 * self._huber_delta_force
                            )
                            huber_elem = xp.where(
                                abs_e <= self._huber_delta_force, quad, lin
                            )  # [nf, nloc, 3]
                            huber_ncomp = 3
                        else:
                            norm_2d = xp.reshape(
                                xp.linalg.vector_norm(
                                    xp.reshape(diff_f_3d, (-1, 3)), axis=1
                                ),
                                _node_shape,
                            )
                            abs_n = norm_2d
                            quad_n = 0.5 * xp.square(norm_2d)
                            lin_n = self._huber_delta_force * (
                                abs_n - 0.5 * self._huber_delta_force
                            )
                            huber_elem = xp.reshape(
                                xp.where(
                                    abs_n <= self._huber_delta_force, quad_n, lin_n
                                ),
                                (*_node_shape, 1),
                            )
                            huber_ncomp = 1
                        l_huber_masked = masked_atom_mean(
                            huber_elem, maskf, huber_ncomp
                        )
                        loss += pref_f * l_huber_masked
                    more_loss["rmse_f"] = self.display_if_exist(
                        xp.sqrt(l2_force_masked), find_force
                    )
                else:
                    if not self.use_huber:
                        loss += pref_f * l2_force_loss
                    else:
                        if not self.f_use_norm:
                            l_huber_loss = _huber_from_residual(
                                diff_f,
                                delta=self._huber_delta_force,
                            )
                        else:
                            force_diff_3 = xp.reshape(diff_f, (-1, 3))
                            force_diff_norm = xp.reshape(
                                xp.linalg.vector_norm(force_diff_3, axis=1), (-1, 1)
                            )
                            l_huber_loss = _huber_from_residual(
                                force_diff_norm,
                                delta=self._huber_delta_force,
                            )
                        loss += pref_f * l_huber_loss
                    more_loss["rmse_f"] = self.display_if_exist(
                        xp.sqrt(l2_force_loss), find_force
                    )
            elif self.loss_func == "mae":
                if maskf is not None:
                    diff_f_3d = xp.reshape(diff_f, (*_node_shape, 3))
                    if not self.f_use_norm:
                        l1_force_masked = masked_atom_mean(xp.abs(diff_f_3d), maskf, 3)
                    else:
                        norm_2d = xp.reshape(
                            xp.linalg.vector_norm(
                                xp.reshape(diff_f_3d, (-1, 3)), axis=1
                            ),
                            _node_shape,
                        )
                        # One L2 norm per atom, hence one label per atom.
                        l1_force_masked = masked_atom_mean(
                            xp.reshape(norm_2d, (*_node_shape, 1)), maskf, 1
                        )
                    loss += pref_f * l1_force_masked
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_masked, find_force
                    )
                else:
                    if not self.f_use_norm:
                        l1_force_loss = xp.mean(xp.abs(diff_f))
                    else:
                        force_diff_3 = xp.reshape(diff_f, (-1, 3))
                        l1_force_loss = xp.mean(
                            xp.linalg.vector_norm(force_diff_3, axis=1)
                        )
                    loss += pref_f * l1_force_loss
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss, find_force
                    )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for force loss."
                )
            if mae:
                if maskf is not None:
                    diff_f_3d = xp.reshape(diff_f, (*_node_shape, 3))
                    mae_f = masked_atom_mean(xp.abs(diff_f_3d), maskf, 3)
                else:
                    mae_f = xp.mean(xp.abs(diff_f))
                more_loss["mae_f"] = self.display_if_exist(mae_f, find_force)
        if self.has_v:
            virial_reshape = xp.reshape(virial, (-1,))
            virial_hat_reshape = xp.reshape(virial_hat, (-1,))
            if self.loss_func == "mse":
                l2_virial_loss = xp.mean(
                    xp.square(virial_hat_reshape - virial_reshape),
                )
                if inv is not None:
                    # Idiom 2 (extensive, k=9): per-frame normalization.
                    v2d = xp.reshape(virial, (_nf, 9))
                    v_hat_2d = xp.reshape(virial_hat, (_nf, 9))
                    se_v = xp.square(v_hat_2d - v2d)  # [nf, 9]
                    per_frame_v = per_frame_component_mean(se_v)  # [nf]
                    if not self.use_huber:
                        loss += pref_v * xp.mean(per_frame_v * inv**norm_exp)
                    else:
                        inv_col = xp.reshape(inv, (_nf, 1))  # [nf, 1]
                        l_huber_v = custom_huber_loss(
                            inv_col * v2d,
                            inv_col * v_hat_2d,
                            delta=self._huber_delta_virial,
                        )
                        loss += pref_v * l_huber_v
                    more_loss["rmse_v"] = self.display_if_exist(
                        xp.sqrt(xp.mean(per_frame_v * inv**2)), find_virial
                    )
                else:
                    if not self.use_huber:
                        loss += atom_norm**norm_exp * (pref_v * l2_virial_loss)
                    else:
                        l_huber_loss = custom_huber_loss(
                            atom_norm * virial_reshape,
                            atom_norm * virial_hat_reshape,
                            delta=self._huber_delta_virial,
                        )
                        loss += pref_v * l_huber_loss
                    more_loss["rmse_v"] = self.display_if_exist(
                        xp.sqrt(l2_virial_loss) * atom_norm, find_virial
                    )
            elif self.loss_func == "mae":
                l1_virial_loss = xp.mean(xp.abs(virial_hat_reshape - virial_reshape))
                if inv is not None:
                    v2d = xp.reshape(virial, (_nf, 9))
                    v_hat_2d = xp.reshape(virial_hat, (_nf, 9))
                    per_frame_v = per_frame_component_mean(
                        xp.abs(v_hat_2d - v2d)
                    )  # [nf]
                    l1_virial_masked = xp.mean(per_frame_v * inv)
                    loss += pref_v * l1_virial_masked
                    more_loss["mae_v"] = self.display_if_exist(
                        l1_virial_masked, find_virial
                    )
                else:
                    loss += atom_norm * (pref_v * l1_virial_loss)
                    more_loss["mae_v"] = self.display_if_exist(
                        l1_virial_loss * atom_norm, find_virial
                    )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for virial loss."
                )
            if mae:
                if inv is not None:
                    v2d = xp.reshape(virial, (_nf, 9))
                    v_hat_2d = xp.reshape(virial_hat, (_nf, 9))
                    per_frame_v = per_frame_component_mean(xp.abs(v_hat_2d - v2d))
                    mae_v = xp.mean(per_frame_v * inv)
                else:
                    mae_v = (
                        xp.mean(xp.abs(virial_hat_reshape - virial_reshape)) * atom_norm
                    )
                more_loss["mae_v"] = self.display_if_exist(mae_v, find_virial)
        if self.has_ae:
            atom_ener_reshape = xp.reshape(atom_ener, (-1,))
            atom_ener_hat_reshape = xp.reshape(atom_ener_hat, (-1,))
            if self.loss_func == "mse":
                l2_atom_ener_loss = xp.mean(
                    xp.square(atom_ener_hat_reshape - atom_ener_reshape),
                )
                if maskf is not None:
                    # Idiom 1 (per-atom masked mean, ncomp=1).
                    ae_2d = xp.reshape(atom_ener, _node_shape)
                    ae_hat_2d = xp.reshape(atom_ener_hat, _node_shape)
                    l2_ae_masked = masked_atom_mean(
                        xp.square(ae_hat_2d - ae_2d)[..., None], maskf, 1
                    )
                    if not self.use_huber:
                        loss += pref_ae * l2_ae_masked
                    else:
                        # Huber applied element-wise then masked-mean.
                        diff_ae = ae_hat_2d - ae_2d
                        abs_ae = xp.abs(diff_ae)
                        quad_ae = 0.5 * xp.square(diff_ae)
                        lin_ae = self._huber_delta_energy * (
                            abs_ae - 0.5 * self._huber_delta_energy
                        )
                        huber_ae = xp.where(
                            abs_ae <= self._huber_delta_energy, quad_ae, lin_ae
                        )
                        l_huber_ae_masked = masked_atom_mean(
                            huber_ae[..., None], maskf, 1
                        )
                        loss += pref_ae * l_huber_ae_masked
                    more_loss["rmse_ae"] = self.display_if_exist(
                        xp.sqrt(l2_ae_masked), find_atom_ener
                    )
                else:
                    if not self.use_huber:
                        loss += pref_ae * l2_atom_ener_loss
                    else:
                        l_huber_loss = custom_huber_loss(
                            atom_ener_reshape,
                            atom_ener_hat_reshape,
                            delta=self._huber_delta_energy,
                        )
                        loss += pref_ae * l_huber_loss
                    more_loss["rmse_ae"] = self.display_if_exist(
                        xp.sqrt(l2_atom_ener_loss), find_atom_ener
                    )
            elif self.loss_func == "mae":
                l1_atom_ener_loss = xp.mean(
                    xp.abs(atom_ener_hat_reshape - atom_ener_reshape)
                )
                if maskf is not None:
                    ae_2d = xp.reshape(atom_ener, _node_shape)
                    ae_hat_2d = xp.reshape(atom_ener_hat, _node_shape)
                    l1_ae_masked = masked_atom_mean(
                        xp.abs(ae_hat_2d - ae_2d)[..., None], maskf, 1
                    )
                    loss += pref_ae * l1_ae_masked
                    more_loss["mae_ae"] = self.display_if_exist(
                        l1_ae_masked, find_atom_ener
                    )
                else:
                    loss += pref_ae * l1_atom_ener_loss
                    more_loss["mae_ae"] = self.display_if_exist(
                        l1_atom_ener_loss, find_atom_ener
                    )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atomic energy loss."
                )
        if self.has_pf:
            atom_pref_reshape = xp.reshape(atom_pref, (-1,))

            if self.loss_func == "mse":
                l2_pref_force_loss = xp.mean(
                    xp.multiply(xp.square(diff_f), atom_pref_reshape),
                )
                if maskf is not None:
                    # Idiom 1 with pref weight (ncomp=3).
                    diff_f_3d = xp.reshape(diff_f, (*_node_shape, 3))
                    pf_3d = xp.reshape(atom_pref, (*_node_shape, 3))
                    l2_pf_masked = masked_atom_mean(
                        xp.square(diff_f_3d) * pf_3d, maskf, 3
                    )
                    loss += pref_pf * l2_pf_masked
                    more_loss["rmse_pf"] = self.display_if_exist(
                        xp.sqrt(l2_pf_masked), effective_find_pf
                    )
                else:
                    loss += pref_pf * l2_pref_force_loss
                    more_loss["rmse_pf"] = self.display_if_exist(
                        xp.sqrt(l2_pref_force_loss), effective_find_pf
                    )
            elif self.loss_func == "mae":
                l1_pref_force_loss = xp.mean(
                    xp.multiply(xp.abs(diff_f), atom_pref_reshape)
                )
                if maskf is not None:
                    diff_f_3d = xp.reshape(diff_f, (*_node_shape, 3))
                    pf_3d = xp.reshape(atom_pref, (*_node_shape, 3))
                    l1_pf_masked = masked_atom_mean(xp.abs(diff_f_3d) * pf_3d, maskf, 3)
                    loss += pref_pf * l1_pf_masked
                    more_loss["mae_pf"] = self.display_if_exist(
                        l1_pf_masked, effective_find_pf
                    )
                else:
                    loss += pref_pf * l1_pref_force_loss
                    more_loss["mae_pf"] = self.display_if_exist(
                        l1_pref_force_loss, effective_find_pf
                    )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atom prefactor force loss."
                )
        if self.has_gf:
            if is_ragged:
                # ``natoms`` below is one number for the whole batch, which a
                # padded batch can honour and a concatenated one cannot: its
                # frames differ in atom count, so ``drdq``, stored per frame
                # against a common atom axis, has no shape to take.
                raise NotImplementedError(
                    "the generalized force loss requires every frame of a "
                    "batch to hold the same number of atoms; a batch whose "
                    "frames are concatenated cannot provide the common atom "
                    "axis its ``drdq`` label is stored against"
                )
            find_drdq = label_dict["find_drdq"]
            drdq = label_dict["drdq"]
            effective_find_gf = find_force * find_drdq
            pref_gf = effective_find_gf * (
                self.limit_pref_gf
                + (self.start_pref_gf - self.limit_pref_gf) * lr_ratio
            )
            if maskf is not None:
                # Mask per-atom forces before projecting onto generalized coords
                # so ghost atoms don't contribute to the generalized force.
                force_3d = xp.reshape(force, (_nf, _nloc, 3))
                force_hat_3d = xp.reshape(force_hat, (_nf, _nloc, 3))
                maskf_col = xp.reshape(maskf, (_nf, _nloc, 1))
                masked_f = force_3d * maskf_col  # [nf, nloc, 3]
                masked_f_hat = force_hat_3d * maskf_col  # [nf, nloc, 3]
                f_flat = xp.reshape(masked_f, (_nf, _nloc * 3))
                f_hat_flat = xp.reshape(masked_f_hat, (_nf, _nloc * 3))
                drdq_reshape = xp.reshape(
                    drdq, (_nf, _nloc * 3, self.numb_generalized_coord)
                )
                gen_force = xp.sum(drdq_reshape * f_flat[:, :, None], axis=1)
                gen_force_hat = xp.sum(drdq_reshape * f_hat_flat[:, :, None], axis=1)
            else:
                force_reshape_nframes = xp.reshape(force, (-1, natoms * 3))
                force_hat_reshape_nframes = xp.reshape(force_hat, (-1, natoms * 3))
                drdq_reshape = xp.reshape(
                    drdq, (-1, natoms * 3, self.numb_generalized_coord)
                )
                gen_force_hat = xp.sum(
                    drdq_reshape * force_hat_reshape_nframes[:, :, None], axis=1
                )
                gen_force = xp.sum(
                    drdq_reshape * force_reshape_nframes[:, :, None], axis=1
                )
            # "bij,bi->bj" einsum replaced with array-API-compatible ops
            diff_gen_force = gen_force_hat - gen_force
            l2_gen_force_loss = xp.mean(xp.square(diff_gen_force))
            loss += pref_gf * l2_gen_force_loss
            more_loss["rmse_gf"] = self.display_if_exist(
                xp.sqrt(l2_gen_force_loss), effective_find_gf
            )
        hessian = model_dict.get("hessian", model_dict.get("energy_derv_r_derv_r"))
        if self.has_h and hessian is not None and "hessian" in label_dict:
            if is_ragged:
                raise NotImplementedError(
                    "the hessian loss requires a rectangular atom axis"
                )
            find_hessian = label_dict.get("find_hessian", 0.0)
            if maskf is not None:
                hessian_shape = (_nf, _nloc * 3, _nloc * 3)
                diff_h = xp.reshape(label_dict["hessian"], hessian_shape) - xp.reshape(
                    hessian, hessian_shape
                )
                # A Hessian element couples two Cartesian atom components, so
                # it is valid only when both corresponding atoms are real.
                l2_hessian_loss = masked_pair_mean(xp.square(diff_h), maskf, ncomp=3)
            else:
                diff_h = xp.reshape(label_dict["hessian"], (-1,)) - xp.reshape(
                    hessian,
                    (-1,),
                )
                l2_hessian_loss = xp.mean(xp.square(diff_h))
            mae_h = None
            if self.loss_func == "mae" or mae:
                if maskf is not None:
                    mae_h = masked_pair_mean(xp.abs(diff_h), maskf, ncomp=3)
                else:
                    mae_h = xp.mean(xp.abs(diff_h))
            if self.loss_func == "mse":
                loss += pref_h * find_hessian * l2_hessian_loss
            elif self.loss_func == "mae":
                loss += pref_h * find_hessian * mae_h
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for hessian loss."
                )
            more_loss["rmse_h"] = self.display_if_exist(
                xp.sqrt(l2_hessian_loss), find_hessian
            )
            if mae:
                more_loss["mae_h"] = self.display_if_exist(mae_h, find_hessian)

        self.l2_l = loss
        more_loss["rmse"] = xp.sqrt(loss)
        self.l2_more = more_loss
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement: list[DataRequirementItem] = []
        if self.has_e:
            label_requirement.append(
                DataRequirementItem(
                    "energy",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_f or self.has_pf or self.relative_f is not None or self.has_gf:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_v:
            label_requirement.append(
                DataRequirementItem(
                    "virial",
                    ndof=9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ae:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_pf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                    default=1.0,
                    source_policy="default" if self.use_default_pf else "tracked",
                )
            )
        if self.has_gf > 0:
            label_requirement.append(
                DataRequirementItem(
                    "drdq",
                    ndof=self.numb_generalized_coord * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.enable_atom_ener_coeff:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener_coeff",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    default=1.0,
                    source_policy="default",
                )
            )
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=False,
                    special_shape="hessian",
                )
            )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        data = {
            "@class": "EnergyLoss",
            # Version 5 identifies the opt-in Hessian fields. Keep ordinary
            # energy losses at version 4 so readers that already support the
            # standard schema remain interoperable across backends.
            "@version": 5 if self.has_h else 4,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "start_pref_pf": self.start_pref_pf,
            "limit_pref_pf": self.limit_pref_pf,
            "relative_f": self.relative_f,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "start_pref_gf": self.start_pref_gf,
            "limit_pref_gf": self.limit_pref_gf,
            "numb_generalized_coord": self.numb_generalized_coord,
            "use_huber": self.use_huber,
            "huber_delta": self.huber_delta,
            "loss_func": self.loss_func,
            "f_use_norm": self.f_use_norm,
            "use_default_pf": self.use_default_pf,
            "intensive_ener_virial": self.intensive_ener_virial,
        }
        if self.has_h:
            # Keep the established cross-backend serialization unchanged for
            # ordinary energy losses; Hessian-only fields are an opt-in schema.
            data["start_pref_h"] = self.start_pref_h
            data["limit_pref_h"] = self.limit_pref_h
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "Loss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        Loss
            The deserialized loss module
        """
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 5, 1)
        data.pop("@class")
        # Backward compatibility: version 1-2 used legacy normalization
        if version < 3:
            data.setdefault("intensive_ener_virial", False)
        # Version 5 introduced explicit Hessian prefactors. Older payloads
        # represent an ordinary energy loss unless these development fields
        # were already present.
        if version < 5:
            data.setdefault("start_pref_h", 0.0)
            data.setdefault("limit_pref_h", 0.0)
        return cls(**data)
