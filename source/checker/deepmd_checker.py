# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    ClassVar,
)

from astroid.nodes.node_classes import (
    Attribute,
    Name,
)
from pylint.checkers import (
    BaseChecker,
)

if TYPE_CHECKING:
    from pylint.lint import (
        PyLinter,
    )


class DPChecker(BaseChecker):
    name = "deepmd-checker"
    msgs: ClassVar[dict] = {
        "E8001": (
            "No explicit device.",
            "no-explicit-device",
            "Require explicit device when initializing a PyTorch tensor.",
        ),
        "E8002": (
            "No explicit dtype.",
            "no-explicit-dtype",
            "Require explicit dtype when initializing a NumPy array, a TensorFlow tensor, or a PyTorch tensor.",
        ),
    }

    def visit_call(self, node) -> None:
        if (
            isinstance(node.func, Attribute)
            and isinstance(node.func.expr, Name)
            and node.func.expr.name in {"np", "tf", "torch", "xp", "jnp"}
            and node.func.attrname
            in {
                # https://pytorch.org/docs/stable/torch.html#creation-ops
                "tensor",
                "zeros",
                "ones",
                "arange",
                "range",
                "empty",
                "full",
                "rand",
                "eye",
                "linspace",
            }
        ):
            no_device = True
            no_dtype = True
            for kw in node.keywords:
                if kw.arg == "device":
                    no_device = False
                if kw.arg == "dtype":
                    no_dtype = False
            if no_device and node.func.expr.name == "torch":
                # only PT needs device
                self.add_message("no-explicit-device", node=node)
            if no_dtype:
                # note: use torch.from_numpy instead of torch.tensor when
                # converting numpy array to tensor
                self.add_message("no-explicit-dtype", node=node)


def register(linter: "PyLinter") -> None:
    pass


def load_configuration(linter) -> None:
    linter.register_checker(DPChecker(linter))
