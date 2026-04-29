# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from importlib.util import (
    find_spec,
)
from typing import (
    TYPE_CHECKING,
    ClassVar,
)

from deepmd.backend.backend import (
    Backend,
)

if TYPE_CHECKING:
    from argparse import (
        Namespace,
    )

    from deepmd.infer.deep_eval import (
        DeepEvalBackend,
    )
    from deepmd.utils.neighbor_stat import (
        NeighborStat,
    )


@Backend.register("pt-expt")
@Backend.register("pytorch-exportable")
class PyTorchExportableBackend(Backend):
    """PyTorch exportable backend."""

    name = "PyTorch-Exportable"
    """The formal name of the backend."""
    features: ClassVar[Backend.Feature] = (
        Backend.Feature.ENTRY_POINT
        | Backend.Feature.DEEP_EVAL
        | Backend.Feature.NEIGHBOR_STAT
        | Backend.Feature.IO
    )
    """The features of the backend."""
    suffixes: ClassVar[list[str]] = [".pte", ".pt2"]
    """The suffixes of the backend."""

    @classmethod
    def match_filename(cls, filename: str) -> int:
        """Recognise pt_expt-trained `.pt` checkpoints in addition to `.pt2`/`.pte`.

        Returns
        -------
        - 1 for the regular `.pte` / `.pt2` suffixes (default behaviour).
        - 2 for `.pt` files whose state-dict uses pt_expt's dpmodel
            parameter naming (`.w`/`.b`); this outranks the legacy pt
            backend's default suffix score (1) so pt_expt-trained `.pt`
            checkpoints route here, while genuine pt-trained `.pt` files
            (which use `.matrix`/`.bias`) keep going to the pt backend.
        - 0 otherwise.
        """
        score = super().match_filename(filename)
        if score:
            return score
        fname = str(filename).lower()
        if not fname.endswith(".pt"):
            return 0
        try:
            import torch

            # weights_only=True avoids unpickling arbitrary code from an
            # untrusted .pt — sniffing only needs the dict keys.
            sd = torch.load(filename, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            keys = list(sd.keys()) if hasattr(sd, "keys") else []
            has_pt_expt = any(k.endswith(".w") or k.endswith(".b") for k in keys)
            has_pt = any(k.endswith(".matrix") or k.endswith(".bias") for k in keys)
            if has_pt_expt and not has_pt:
                return 2
        except Exception:
            pass
        return 0

    def is_available(self) -> bool:
        """Check if the backend is available.

        Returns
        -------
        bool
            Whether the backend is available.
        """
        return find_spec("torch") is not None

    @property
    def entry_point_hook(self) -> Callable[["Namespace"], None]:
        """The entry point hook of the backend.

        Returns
        -------
        Callable[[Namespace], None]
            The entry point hook of the backend.
        """
        from deepmd.pt_expt.entrypoints.main import main as deepmd_main

        return deepmd_main

    @property
    def deep_eval(self) -> type["DeepEvalBackend"]:
        """The Deep Eval backend of the backend.

        Returns
        -------
        type[DeepEvalBackend]
            The Deep Eval backend of the backend.
        """
        from deepmd.pt_expt.infer.deep_eval import (
            DeepEval,
        )

        return DeepEval

    @property
    def neighbor_stat(self) -> type["NeighborStat"]:
        """The neighbor statistics of the backend.

        Returns
        -------
        type[NeighborStat]
            The neighbor statistics of the backend.
        """
        from deepmd.pt_expt.utils.neighbor_stat import (
            NeighborStat,
        )

        return NeighborStat

    @property
    def serialize_hook(self) -> Callable[[str], dict]:
        """The serialize hook to convert the model file to a dictionary.

        Returns
        -------
        Callable[[str], dict]
            The serialize hook of the backend.
        """
        from deepmd.pt_expt.utils.serialization import (
            serialize_from_file,
        )

        return serialize_from_file

    @property
    def deserialize_hook(self) -> Callable[[str, dict], None]:
        """The deserialize hook to convert the dictionary to a model file.

        Returns
        -------
        Callable[[str, dict], None]
            The deserialize hook of the backend.
        """
        from deepmd.pt_expt.utils.serialization import (
            deserialize_to_file,
        )

        return deserialize_to_file
