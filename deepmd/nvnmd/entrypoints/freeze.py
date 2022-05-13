
from deepmd.env import tf
from deepmd.nvnmd.utils.fio import FioDic

def filter_tensorVariableList(tensorVariableList) -> dict:
    """
    descrpt_attr/t_avg:0
    descrpt_attr/t_std:0
    filter_type_{atom i}/matrix_{layer l}_{atomj}:0
    filter_type_{atom i}/bias_{layer l}_{atomj}:0
    layer_{layer l}_type_{atom i}/matrix:0
    layer_{layer l}_type_{atom i}/bias:0
    final_layer_type_{atom i}/matrix:0
    final_layer_type_{atom i}/bias:0
    """
    nameList = [tv.name for tv in tensorVariableList]
    nameList = [name.replace(':0','') for name in nameList]
    nameList = [name.replace('/','.') for name in nameList]

    dic_name_tv = {}
    for ii in range(len(nameList)):
        name = nameList[ii]
        tv = tensorVariableList[ii]
        if (name.startswith('descrpt_attr') or \
            name.startswith('filter_type_') or \
            name.startswith('layer_') or \
            name.startswith('final_layer_type_') ) and \
            ('Adam' not in name) and \
            ('XXX' not in name):
            dic_name_tv[name] = tv
    
    return dic_name_tv

def save_weight(sess, file_name: str='nvnmd/weight.npy'):
    tvs = tf.global_variables()
    dic_key_tv = filter_tensorVariableList(tvs)
    dic_key_value = {}
    for key in dic_key_tv.keys():
        value = sess.run(dic_key_tv[key])
        dic_key_value[key] = value
    
    FioDic().save(file_name, dic_key_value)

     







