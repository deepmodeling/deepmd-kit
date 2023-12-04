from typing import List

from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import paddle
from deepmd.env import tf


class Spin(paddle.nn.Layer):
    """Class for spin.

    Parameters
    ----------
    use_spin
                Whether to use atomic spin model for each atom type
    spin_norm
                The magnitude of atomic spin for each atom type with spin
    virtual_len
                The distance between virtual atom representing spin and its corresponding real atom for each atom type with spin
    """

    def __init__(
        self,
        use_spin: List[bool] = None,
        spin_norm: List[float] = None,
        virtual_len: List[float] = None,
    ) -> None:
        super().__init__()
        """Constructor."""
        self.use_spin = use_spin
        self.spin_norm = spin_norm
        self.virtual_len = virtual_len
        self.ntypes_spin = self.use_spin.count(True)
        self.register_buffer(
            "buffer_ntypes_spin",
            paddle.to_tensor([self.ntypes_spin], dtype="int32"),
        )
        self.register_buffer(
            "buffer_virtual_len",
            paddle.to_tensor([self.virtual_len], dtype=paddle.get_default_dtype()),
        )
        self.register_buffer(
            "buffer_spin_norm",
            paddle.to_tensor([self.spin_norm], dtype=paddle.get_default_dtype()),
        )

    def get_ntypes_spin(self) -> int:
        """Returns the number of atom types which contain spin."""
        return self.ntypes_spin

    def get_use_spin(self) -> List[bool]:
        """Returns the list of whether to use spin for each atom type."""
        return self.use_spin

    def get_spin_norm(self) -> List[float]:
        """Returns the list of magnitude of atomic spin for each atom type."""
        return self.spin_norm

    def get_virtual_len(self) -> List[float]:
        """Returns the list of distance between real atom and virtual atom for each atom type."""
        return self.virtual_len
