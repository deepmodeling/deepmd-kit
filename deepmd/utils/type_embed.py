import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.utils.network import one_layer
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net_type, embedding_net,share_embedding_network_oneside,share_embedding_network_twoside

import math
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter, get_np_precision
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.tabulate import DeepTabulate

class Type_embed_net():
    def __init__(self,
                ntypes: int,
                type_filter: List[int]=[],
                resnet_dt: bool = False,
                seed: int = 1,
                activation_function: str = 'tanh',
                precision: str = 'default',
    )->None:
        self.ntypes = ntypes
        self.type_filter = type_filter
        self.seed = seed
        self.filter_resnet_dt = resnet_dt
        self.filter_precision = get_precision(precision)
        self.filter_np_precision = get_np_precision(precision)
        self.filter_activation_fn = get_activation_func(activation_function)


    def _type_embed(self, 
                    atype,
                    reuse = None, 
                    suffix = '',
                    trainable = True):
        
        ebd_type = tf.cast(tf.one_hot(tf.cast(atype,dtype=tf.int32),int(self.ntypes)), self.filter_precision)
        ebd_type = tf.reshape(ebd_type, [-1, self.ntypes])
        name = 'type_embed_net_' 
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
          ebd_type = embedding_net_type(ebd_type,
                                 [self.ntypes]+self.type_filter,
                                 activation_fn = self.filter_activation_fn,
                                 precision = self.filter_precision,
                                 resnet_dt = self.filter_resnet_dt,
                                 seed = self.seed,
                                 trainable = trainable)

        ebd_type = tf.reshape(ebd_type, [-1, self.type_filter[-1]]) # nnei * type_filter[-1]
        return ebd_type            


    def fetch_type_embedding(self) -> tf.Tensor:
        _atom_type_ = []
        for ii in range(self.ntypes):
            _atom_type_.append(ii)
        _atom_type_ = tf.convert_to_tensor(_atom_type_,dtype = GLOBAL_TF_FLOAT_PRECISION)
        type_embedding = self._type_embed( tf.one_hot(tf.cast(_atom_type_,dtype=tf.int32),int(self.ntypes)),
                                        reuse = True,
                                        suffix = '',
                                        trainable = True) 
        return type_embedding