# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt Wigner-D calculator with an opt-in fused Triton monomial fast path.

The dpmodel :class:`WignerDCalculator` is array-API only and evaluates the
degree ``l >= 2`` monomial design matrices through the dense power-table chain.
This wrapper injects the reference pt inference fast path around the two
monomial hot paths -- the shared ``l >= 3`` kernel and the ``l = 2`` degree-4
contraction -- mirroring ``deepmd.pt.model.descriptor.sezm_nn.wignerd``.

The fused monomial operator is sourced from the central
:mod:`deepmd.kernels.triton.sezm.wigner_monomials` package and gated by the
integer inference level ``DP_TRITON_INFER`` (see
:func:`deepmd.kernels.utils.triton_infer_level`); the fast path requires level
``>= 1``.  It runs only during inference (``not self.training``) on CUDA, and
the operator self-guards Triton availability and falls back to an eager
reference off CUDA / on fp64, so importing this module is safe on CPU-only
environments; training and CPU / fp64 inference use the dpmodel dense path.
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
from deepmd.kernels.utils import (
    triton_infer_level,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class WignerDCalculator(WignerDCalculatorDP):
    """Wigner-D calculator with an opt-in fused Triton monomial inference path."""

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(lmax, eps=eps, precision=precision)
        # Inference fast-path gate (``DP_TRITON_INFER >= 1``): read once at
        # construction so it is a compile-time constant in the traced
        # (``make_fx``) graph, and it only takes effect during inference.
        self._use_triton_monomials = triton_infer_level() >= 1
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
        :mod:`deepmd.kernels.triton.sezm.wigner_monomials`); construction-time
        solves and CPU targets keep the dense power-table chain.
        """
        exps = self._monomial_exponents_flat.get(exp_name)
        if (
            self._use_triton_monomials
            and exps is not None
            and edge_quaternion.is_cuda
            and not self.training
        ):
            from deepmd.kernels.triton.sezm.wigner_monomials import (
                wigner_monomials,
            )

            return wigner_monomials(edge_quaternion, exps, max_power)
        return super()._monomial_matrix(edge_quaternion, exp_name, max_power)

    def _compute_l2_block(self, edge_quaternion: torch.Tensor) -> torch.Tensor:
        """Compute the ``l=2`` block from the degree-4 quaternion contraction.

        The fused inference path collapses the 256 rank-4 index tuples onto
        the 35 unique degree-4 monomials, replacing the ``(E, 4, 4, 4, 4)``
        outer product with a monomial evaluation and one ``(E, 35) x (35, 25)``
        product with no large intermediate.
        """
        exps = self._monomial_exponents_flat.get("exp_l2")
        if (
            self._use_triton_monomials
            and exps is not None
            and edge_quaternion.is_cuda
            and not self.training
        ):
            from deepmd.kernels.triton.sezm.wigner_monomials import (
                wigner_monomials,
            )

            monomials = wigner_monomials(edge_quaternion, exps, 4)
            # ``_l2_monomial_coeff`` is stored as the fp64 dpmodel constant; cast
            # it to the monomial dtype so the fused fp32 path multiplies operands
            # of one dtype (mirrors the base's runtime cast of the Wigner
            # constants to the edge dtype).
            coeff = self._l2_monomial_coeff.to(monomials.dtype)
            return torch.matmul(monomials, coeff).view(-1, 5, 5)
        return super()._compute_l2_block(edge_quaternion)


# WignerDCalculator.deserialize raises NotImplementedError by design (its
# tables are derived constants); rebuild from the stored constructor args.
register_dpmodel_mapping(
    WignerDCalculatorDP,
    lambda v: WignerDCalculator(v.lmax, eps=v.eps, precision=v.precision),
)
