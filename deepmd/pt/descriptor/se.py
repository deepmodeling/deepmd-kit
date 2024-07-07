import re
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.env import (
    EMBEDDING_NET_PATTERN,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

import torch


class DescrptSe():
    """A base class for smooth version of descriptors.

    Notes
    -----
    All of these descriptors have an environmental matrix and an
    embedding network (:meth:`deepmd.tf.utils.network.embedding_net`), so
    they can share some similiar methods without defining them twice.

    Attributes
    ----------
    embedding_net_variables : dict
        initial embedding network variables
    descrpt_reshape : tf.Tensor
        the reshaped descriptor
    descrpt_deriv : tf.Tensor
        the descriptor derivative
    rij : tf.Tensor
        distances between two atoms
    nlist : tf.Tensor
        the neighbor list

    """

    def _identity_tensors(self, suffix: str = "") -> None:
        """Identify tensors which are expected to be stored and restored.

        Notes
        -----
        These tensors will be indentitied:
            self.descrpt_reshape : o_rmat
            self.descrpt_deriv : o_rmat_deriv
            self.rij : o_rij
            self.nlist : o_nlist
        Thus, this method should be called during building the descriptor and
        after these tensors are initialized.

        Parameters
        ----------
        suffix : str
            The suffix of the scope
        """