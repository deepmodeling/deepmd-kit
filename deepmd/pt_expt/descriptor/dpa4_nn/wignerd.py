# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt Wigner-D calculator with an opt-in accelerated monomial fast path.

The dpmodel :class:`WignerDCalculator` is array-API only and evaluates the
degree ``l >= 2`` monomial design matrices through the dense power-table chain.
This wrapper injects the reference pt inference fast path around the two
monomial hot paths -- the shared ``l >= 3`` kernel and the ``l = 2`` degree-4
contraction -- mirroring ``deepmd.pt.model.descriptor.sezm_nn.wignerd``.

The monomial operator is supplied by the selected Triton or cuTile inference
backend. It runs only during inference (``not self.training``) on CUDA; training
and CPU inference use the dpmodel dense path.
"""

from __future__ import (
    annotations,
)

from itertools import (
    product,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerSmallOrderCoefficients,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    triton_infer_level,
    use_cutile_infer,
)

# Prefix under which the low-order polynomial kernels are held as buffers of
# the calculator; the container is pointed at them (see
# ``_adopt_small_order_kernels``).
_SMALL_ORDER_PREFIX = "_small_order_"

# Highest degree the container defines a specialized kernel for; its name set
# saturates there.
_MAX_SUPPORTED_LMAX = 10


def _small_order_buffer_names() -> tuple[str, ...]:
    """Buffer names of every low-order kernel the container can hold.

    Queried from the container over the supported degree range rather than
    listed a second time, so a kernel added there is covered here without an
    edit. ``required_kernel_names`` is cumulative in ``lmax``, so the largest
    degree yields the complete set.
    """
    names = WignerSmallOrderCoefficients.required_kernel_names(_MAX_SUPPORTED_LMAX)
    return tuple(f"{_SMALL_ORDER_PREFIX}{name}" for name in names)


@torch_module
class WignerDCalculator(WignerDCalculatorDP):
    """Wigner-D calculator with an opt-in accelerated monomial inference path."""

    # Every array below is a pure function of ``lmax``, rebuilt by ``__init__``.
    # Declaring them keeps them out of the state dict, which both leaves stored
    # checkpoints loadable and stops a checkpoint from overriding a value the
    # configuration determines.
    CONFIG_DERIVED_ARRAYS = ("_l2_monomial_coeff", *_small_order_buffer_names())

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(lmax, eps=eps, precision=precision)
        if self.lmax >= 2:
            # Flatten the monomial exponent tables to Python constants in
            # eager context: the fused monomial operator bakes them into the
            # kernel at compile time, and a trace-time ``.tolist()`` would
            # create unbacked symbols under ``make_fx`` and abort export.
            self._monomial_exponents_flat: dict[str, list[int]] = {}
            for exp_name in ("exp_l3", "exp_l4", "exp_l5", "exp_l6"):
                exps = getattr(self.small_order_kernels, exp_name, None)
                if exps is not None:
                    self._monomial_exponents_flat[exp_name] = [
                        int(v) for v in exps.reshape(-1).tolist()
                    ]
            # The monomial basis routes through whichever accelerated backend
            # is selected; the two gates are mutually exclusive.
            self._use_cutile_monomials = use_cutile_infer()
            self._use_triton_monomials = triton_infer_level() >= 1
            # The l = 2 contraction tensor collapsed onto the 35 unique
            # degree-4 monomials: column m of the coefficient matrix sums
            # C_l2[:, :, p] over the 4^4 index tuples p whose component
            # multiplicities equal the monomial exponents.
            exp_l2: list[int] = []
            columns: list[np.ndarray] = []
            index_of: dict[tuple[int, int, int, int], int] = {}
            c_l2 = self.small_order_kernels.C_l2
            for p in product(range(4), repeat=4):
                counts = (p.count(0), p.count(1), p.count(2), p.count(3))
                if counts not in index_of:
                    index_of[counts] = len(index_of)
                    exp_l2.extend(counts)
                    columns.append(np.zeros_like(c_l2[:, :, 0, 0, 0, 0]))
                columns[index_of[counts]] = (
                    columns[index_of[counts]] + c_l2[:, :, p[0], p[1], p[2], p[3]]
                )
            self._monomial_exponents_flat["exp_l2"] = exp_l2
            # Assigned as a numpy array so ``dpmodel_setattr`` registers it as a
            # torch buffer (fp64, matching the other dpmodel Wigner constants).
            self._l2_monomial_coeff = np.stack([c.reshape(-1) for c in columns], axis=0)
        # Adopted after the NumPy construction above, which consumes ``C_l2``.
        self._adopt_small_order_kernels()

    def _adopt_small_order_kernels(self) -> None:
        """Register the low-order polynomial kernels as buffers of this module.

        The dpmodel calculator holds them as NumPy arrays inside a plain
        container, which the generic conversion cannot see: it inspects the
        module's own attributes, not the contents of an object one of them
        points to. Every evaluation then converts them again, and a NumPy to
        CUDA conversion is a synchronizing host-to-device copy -- three of them
        per step on the deployed degree range, each draining the pipeline.

        Registering each kernel as a buffer moves it to the device once, with
        the module. The container is then pointed at the buffers, so the
        dpmodel evaluation reads a tensor already in the working namespace and
        ``xp_asarray_nodetach`` returns it untouched.
        """
        kernels = getattr(self, "small_order_kernels", None)
        if kernels is None:
            return
        for name in type(kernels).required_kernel_names(self.lmax):
            array = getattr(kernels, name, None)
            if array is None or isinstance(array, torch.Tensor):
                continue
            # Assigning the NumPy array to this module registers it as a
            # buffer (``dpmodel_setattr``); the container then aliases it.
            buffer_name = f"{_SMALL_ORDER_PREFIX}{name}"
            setattr(self, buffer_name, array)
            setattr(kernels, name, getattr(self, buffer_name))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def _monomial_matrix(
        self,
        edge_quaternion: torch.Tensor,
        exp_name: str,
        max_power: int,
    ) -> torch.Tensor:
        """Evaluate one degree kernel's monomial basis, with the fused fast path.

        On the CUDA inference path the fused operator evaluates the monomials
        in registers with the exponent table baked in at compile time (see
        :mod:`.triton.wigner_monomials`); construction-time solves and CPU
        targets keep the dense power-table chain.
        """
        exponents = self._monomial_exponents_flat.get(exp_name)
        if (
            exponents is not None
            and edge_quaternion.is_cuda
            and not self.training
            and (self._use_triton_monomials or self._use_cutile_monomials)
        ):
            if self._use_cutile_monomials:
                from deepmd.pt_expt.kernels.cutile.sezm.wigner_monomials import (
                    wigner_monomials as monomial_basis,
                )
            else:
                from deepmd.pt_expt.kernels.triton.sezm.wigner_monomials import (
                    wigner_monomials as monomial_basis,
                )

            return monomial_basis(edge_quaternion, exponents, max_power)
        return super()._monomial_matrix(edge_quaternion, exp_name, max_power)

    def _compute_l2_block(self, edge_quaternion: torch.Tensor) -> torch.Tensor:
        """Compute the ``l=2`` block from the degree-4 quaternion contraction.

        The fused inference path collapses the 256 rank-4 index tuples onto
        the 35 unique degree-4 monomials, replacing the ``(E, 4, 4, 4, 4)``
        outer product with a monomial evaluation and one ``(E, 35) x (35, 25)``
        product with no large intermediate.
        """
        exponents = self._monomial_exponents_flat.get("exp_l2")
        if (
            exponents is not None
            and edge_quaternion.is_cuda
            and not self.training
            and (self._use_triton_monomials or self._use_cutile_monomials)
        ):
            if self._use_cutile_monomials:
                from deepmd.pt_expt.kernels.cutile.sezm.wigner_monomials import (
                    wigner_monomials as monomial_basis,
                )
            else:
                from deepmd.pt_expt.kernels.triton.sezm.wigner_monomials import (
                    wigner_monomials as monomial_basis,
                )

            monomials = monomial_basis(edge_quaternion, exponents, 4)
            # The dpmodel-derived coefficient stays fp64, so it follows the
            # base calculator's runtime cast to the edge compute dtype.
            D_flat = torch.matmul(
                monomials, self._l2_monomial_coeff.to(monomials.dtype)
            )
            return D_flat.view(edge_quaternion.shape[0], 5, 5)
        return super()._compute_l2_block(edge_quaternion)


# WignerDCalculator.deserialize raises NotImplementedError by design (its
# tables are derived constants); rebuild from the stored constructor args.
register_dpmodel_mapping(
    WignerDCalculatorDP,
    lambda v: WignerDCalculator(v.lmax, eps=v.eps, precision=v.precision),
)
