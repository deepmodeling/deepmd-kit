# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)


class Spin:
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
        use_spin: Optional[List[bool]] = None,
        spin_norm: Optional[List[float]] = None,
        virtual_len: Optional[List[float]] = None,
    ) -> None:
        """Constructor."""
        self.use_spin = use_spin
        self.spin_norm = spin_norm
        self.virtual_len = virtual_len
        self.ntypes_spin = self.use_spin.count(True)

    def build(
        self,
        reuse=None,
        suffix="",
    ):
        """Build the computational graph for the spin.

        Parameters
        ----------
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        embedded_types
            The computational graph for embedded types
        """
        name = "spin_attr" + suffix
        with tf.variable_scope(name, reuse=reuse):
            t_ntypes_spin = tf.constant(
                self.ntypes_spin, name="ntypes_spin", dtype=tf.int32
            )
            t_virtual_len = tf.constant(
                self.virtual_len,
                name="virtual_len",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            t_spin_norm = tf.constant(
                self.spin_norm,
                name="spin_norm",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
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
