# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend helper for `*.pretrained` model aliases."""

from __future__ import (
    annotations,
)

from functools import (
    lru_cache,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.pretrained.download import (
    resolve_model_path,
)

if TYPE_CHECKING:
    import numpy as np

    from deepmd.infer.deep_eval import (
        DeepEval,
        DeepEvalBackend,
    )


def parse_pretrained_alias(model_file: str) -> str:
    """Extract model name from ``*.pretrained`` alias string."""
    alias = Path(model_file).name
    suffix = ".pretrained"
    if not alias.endswith(suffix):
        raise ValueError(f"Invalid pretrained alias: {model_file}")

    model_name = alias[: -len(suffix)]
    if not model_name:
        raise ValueError(f"Invalid pretrained alias: {model_file}")

    return model_name


@lru_cache(maxsize=1)
def get_pretrained_deep_eval_backend() -> type[DeepEvalBackend]:
    """Build and cache the concrete DeepEval adapter lazily."""
    # Avoid circular import when deepmd backend entrypoints are loading.
    from deepmd.infer.deep_eval import (
        DeepEvalBackend,
    )

    class PretrainedDeepEvalBackend(DeepEvalBackend):
        """Resolve alias and delegate to backend selected by resolved model path."""

        def __init__(
            self,
            model_file: str,
            output_def: object,
            *args: object,
            auto_batch_size: object = True,
            neighbor_list: object | None = None,
            **kwargs: object,
        ) -> None:
            model_name = parse_pretrained_alias(model_file)
            resolved = str(resolve_model_path(model_name))

            # DeepEvalBackend.__new__ dispatches by resolved suffix (.pt/.pb/.dp...)
            self._backend = DeepEvalBackend(
                resolved,
                output_def,
                *args,
                auto_batch_size=auto_batch_size,
                neighbor_list=neighbor_list,
                **kwargs,
            )

        def eval(
            self,
            coords: np.ndarray,
            cells: np.ndarray | None,
            atom_types: np.ndarray,
            atomic: bool = False,
            fparam: np.ndarray | None = None,
            aparam: np.ndarray | None = None,
            **kwargs: Any,
        ) -> dict[str, np.ndarray]:
            return self._backend.eval(
                coords,
                cells,
                atom_types,
                atomic,
                fparam=fparam,
                aparam=aparam,
                **kwargs,
            )

        def eval_descriptor(
            self,
            coords: np.ndarray,
            cells: np.ndarray | None,
            atom_types: np.ndarray,
            fparam: np.ndarray | None = None,
            aparam: np.ndarray | None = None,
            efield: np.ndarray | None = None,
            mixed_type: bool = False,
            **kwargs: Any,
        ) -> np.ndarray:
            return self._backend.eval_descriptor(
                coords,
                cells,
                atom_types,
                fparam=fparam,
                aparam=aparam,
                efield=efield,
                mixed_type=mixed_type,
                **kwargs,
            )

        def eval_fitting_last_layer(
            self,
            coords: np.ndarray,
            cells: np.ndarray | None,
            atom_types: np.ndarray,
            fparam: np.ndarray | None = None,
            aparam: np.ndarray | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            return self._backend.eval_fitting_last_layer(
                coords,
                cells,
                atom_types,
                fparam=fparam,
                aparam=aparam,
                **kwargs,
            )

        def get_rcut(self) -> float:
            return self._backend.get_rcut()

        def get_ntypes(self) -> int:
            return self._backend.get_ntypes()

        def get_type_map(self) -> list[str]:
            return self._backend.get_type_map()

        def get_dim_fparam(self) -> int:
            return self._backend.get_dim_fparam()

        def has_default_fparam(self) -> bool:
            return self._backend.has_default_fparam()

        def get_dim_aparam(self) -> int:
            return self._backend.get_dim_aparam()

        @property
        def model_type(self) -> type[DeepEval]:
            return self._backend.model_type

        def get_sel_type(self) -> list[int]:
            return self._backend.get_sel_type()

        def get_numb_dos(self) -> int:
            return self._backend.get_numb_dos()

        def get_has_efield(self) -> bool:
            return self._backend.get_has_efield()

        def get_has_spin(self) -> bool:
            return self._backend.get_has_spin()

        def get_has_hessian(self) -> bool:
            return self._backend.get_has_hessian()

        def get_var_name(self) -> str:
            return self._backend.get_var_name()

        def get_ntypes_spin(self) -> int:
            return self._backend.get_ntypes_spin()

        def get_model(self) -> Any:
            return self._backend.get_model()

    return PretrainedDeepEvalBackend
