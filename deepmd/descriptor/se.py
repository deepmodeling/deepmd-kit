from typing import Tuple, List

from deepmd.env import tf
from deepmd.utils.graph import get_embedding_net_variables, get_tensor_by_name
from .descriptor import Descriptor


class DescrptSe (Descriptor):
    """A base class for smooth version of descriptors.
    
    Notes
    -----
    All of these descriptors have an environmental matrix and an
    embedding network (:meth:`deepmd.utils.network.embedding_net`), so
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
    def _identity_tensors(self, suffix : str = "") -> None:
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
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name = 'o_rmat' + suffix)
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name = 'o_rmat_deriv' + suffix)
        self.rij = tf.identity(self.rij, name = 'o_rij' + suffix)
        self.nlist = tf.identity(self.nlist, name = 'o_nlist' + suffix)

    def get_tensor_names(self, suffix : str = "") -> Tuple[str]:
        """Get names of tensors.
        
        Parameters
        ----------
        suffix : str
            The suffix of the scope

        Returns
        -------
        Tuple[str]
            Names of tensors
        """
        return (f'o_rmat{suffix}:0', f'o_rmat_deriv{suffix}:0', f'o_rij{suffix}:0', f'o_nlist{suffix}:0')

    def pass_tensors_from_frz_model(self,
                                    descrpt_reshape : tf.Tensor,
                                    descrpt_deriv   : tf.Tensor,
                                    rij             : tf.Tensor,
                                    nlist           : tf.Tensor
    ):
        """
        Pass the descrpt_reshape tensor as well as descrpt_deriv tensor from the frz graph_def

        Parameters
        ----------
        descrpt_reshape
                The passed descrpt_reshape tensor
        descrpt_deriv
                The passed descrpt_deriv tensor
        rij
                The passed rij tensor
        nlist
                The passed nlist tensor
        """
        self.rij = rij
        self.nlist = nlist
        self.descrpt_deriv = descrpt_deriv
        self.descrpt_reshape = descrpt_reshape

    def init_variables(self,
                       model_file : str,
                       suffix : str = "",
    ) -> None:
        """
        Init the embedding net variables with the given frozen model

        Parameters
        ----------
        model_file : str
            The input frozen model file
        suffix : str, optional
            The suffix of the scope
        """
        self.embedding_net_variables = get_embedding_net_variables(model_file, suffix = suffix)
        self.davg = get_tensor_by_name(model_file, 'descrpt_attr%s/t_avg' % suffix)
        self.tavg = get_tensor_by_name(model_file, 'descrpt_attr%s/t_std' % suffix)

    @property
    def precision(self) -> tf.DType:
        """Precision of filter network."""
        return self.filter_precision
