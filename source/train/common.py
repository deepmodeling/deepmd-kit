import os,warnings,fnmatch
import numpy as np
import math
from deepmd.env import tf
from deepmd.env import op_module
from deepmd.RunOptions import global_tf_float_precision
import json
import yaml

# def gelu(x):
#     """Gaussian Error Linear Unit.
#     This is a smoother version of the RELU.
#     Original paper: https://arxiv.org/abs/1606.08415
#     Args:
#     x: float Tensor to perform activation.
#     Returns:
#     `x` with the GELU activation applied.
#     """
#     cdf = 0.5 * (1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
#     return x * cdf
def gelu(x):
    return op_module.gelu(x)

data_requirement = {}
activation_fn_dict = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "softplus": tf.nn.softplus,
    "sigmoid": tf.sigmoid,
    "tanh": tf.nn.tanh,
    "gelu": gelu
}
def add_data_requirement(key, 
                         ndof, 
                         atomic = False, 
                         must = False, 
                         high_prec = False,
                         type_sel = None,
                         repeat = 1) :
    data_requirement[key] = {'ndof': ndof, 
                             'atomic': atomic,
                             'must': must, 
                             'high_prec': high_prec,
                             'type_sel': type_sel,
                             'repeat': repeat,
    }
    
def select_idx_map(atom_type, 
                   type_sel):
    sort_type_sel = np.sort(type_sel)
    idx_map = np.array([], dtype = int)
    for ii in sort_type_sel:
        idx_map = np.append(idx_map, np.where(atom_type == ii))
    return idx_map


def make_default_mesh(test_box, cell_size = 3.0) :
    # nframes = test_box.shape[0]
    # default_mesh = np.zeros([nframes, 6], dtype = np.int32)
    # for ff in range(nframes):
    #     ncell = np.ones (3, dtype=np.int32)
    #     for ii in range(3) :
    #         ncell[ii] = int ( np.linalg.norm(test_box[ff][ii]) / cell_size )
    #         if (ncell[ii] < 2) : ncell[ii] = 2
    #     default_mesh[ff][3] = ncell[0]
    #     default_mesh[ff][4] = ncell[1]
    #     default_mesh[ff][5] = ncell[2]
    # return default_mesh
    nframes = test_box.shape[0]
    lboxv = np.linalg.norm(test_box.reshape([-1, 3, 3]), axis = 2)
    avg_lboxv = np.average(lboxv, axis = 0)
    ncell = (avg_lboxv / cell_size).astype(np.int32)
    ncell[ncell < 2] = 2
    default_mesh = np.zeros (6, dtype = np.int32)
    default_mesh[3:6] = ncell
    return default_mesh    


class ClassArg () : 
    def __init__ (self) :
        self.arg_dict = {}
        self.alias_map = {}
    
    def add (self, 
             key,
             types_,
             alias = None,
             default = None, 
             must = False) :
        if type(types_) is not list :
            types = [types_]
        else :
            types = types_
        if alias is not None :
            if type(alias) is not list :
                alias_ = [alias]
            else:
                alias_ = alias
        else :
            alias_ = []

        self.arg_dict[key] = {'types' : types,
                              'alias' : alias_,
                              'value' : default, 
                              'must': must}
        for ii in alias_ :
            self.alias_map[ii] = key

        return self


    def _add_single(self, key, data) :
        vtype = type(data)
        if data is None:
            return data
        if not(vtype in self.arg_dict[key]['types']) :
            # ! altered by Marián Rynik
            # try the type convertion to one of the types
            for tp in self.arg_dict[key]['types']:
                try :
                    vv = tp(data)
                except TypeError:
                    pass
                else:
                    break
            else:
                raise TypeError ("cannot convert provided key \"%s\" to type(s) %s " % (key, str(self.arg_dict[key]['types'])) )
        else :
            vv = data
        self.arg_dict[key]['value'] = vv

    
    def _check_must(self) :
        for kk in self.arg_dict:
            if self.arg_dict[kk]['must'] and self.arg_dict[kk]['value'] is None:
                raise RuntimeError('key \"%s\" must be provided' % kk)


    def parse(self, jdata) :
        for kk in jdata.keys() :
            if kk in self.arg_dict :
                key = kk
                self._add_single(key, jdata[kk])
            else:
                if kk in self.alias_map: 
                    key = self.alias_map[kk]
                    self._add_single(key, jdata[kk])
        self._check_must()
        return self.get_dict()

    def get_dict(self) :
        ret = {}
        for kk in self.arg_dict.keys() :
            ret[kk] = self.arg_dict[kk]['value']
        return ret

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json database must provide key " + key )
    else :
        return jdata[key]

def j_must_have_d (jdata, key, deprecated_key) :
    if not key in jdata.keys() :
        # raise RuntimeError ("json database must provide key " + key )
        for ii in deprecated_key :
            if ii in jdata.keys() :
                warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % (ii,key))
                return jdata[ii]
        raise RuntimeError ("json database must provide key " + key )        
    else :
        return jdata[key]

def j_have (jdata, key) :
    return key in jdata.keys() 

def j_loader(filename):

    if filename.endswith("json"):
        with open(filename, 'r') as fp:
            return json.load(fp)
    elif filename.endswith(("yml", "yaml")):
        with open(filename, 'r') as fp:
            return yaml.safe_load(fp)
    else:
        raise TypeError("config file must be json, or yaml/yml")

def get_activation_func(activation_fn):
    if activation_fn not in activation_fn_dict:
        raise RuntimeError(activation_fn+" is not a valid activation function")
    return activation_fn_dict[activation_fn]

def expand_sys_str(root_dir):
    matches = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, 'type.raw'):
            matches.append(root)
    return matches

def get_precision(precision):
    if precision == "default":
        return  global_tf_float_precision
    elif precision == "float16":
        return tf.float16
    elif precision == "float32":
        return tf.float32
    elif precision == "float64":
        return tf.float64
    else:
        raise RuntimeError("%d is not a valid precision" % precision)

