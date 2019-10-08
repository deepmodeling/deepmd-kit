import os,json,warnings
from deepmd.common import j_have,j_must_have,j_must_have_d

def convert_input_v0_v1(jdata, warning = True, dump = None) :
    output = {}
    if 'with_distrib' in jdata:
        output['with_distrib'] = jdata['with_distrib']
    if jdata['use_smooth'] :
        output['model'] = _smth_model(jdata)
    else:
        output['model'] = _nonsmth_model(jdata)
    output['learning_rate'] = _learning_rate(jdata)
    output['loss'] = _loss(jdata)
    output['training'] = _training(jdata)
    _warnning_input_v0_v1(dump)
    if dump is not None:
       with open(dump, 'w') as fp:
          json.dump(output, fp, indent=4)        
    return output

def _warnning_input_v0_v1(fname) :
    msg = 'It seems that you are using a deepmd-kit input of version 0.x.x, which is deprecated. we have converted the input to >1.0.0 compatible'
    if fname is not None:
        msg += ', and output it to file ' + fname
    warnings.warn(msg)

def _nonsmth_model(jdata):
    model = {}
    model['descriptor'] = _nonsmth_descriptor(jdata)
    model['fitting_net'] = _fitting_net(jdata)
    return model

def _smth_model(jdata):
    model = {}
    model['descriptor'] = _smth_descriptor(jdata)
    model['fitting_net'] = _fitting_net(jdata)
    return model

def _nonsmth_descriptor(jdata) :
    output = {}
    seed = None
    if j_have (jdata, 'seed') :
        seed = jdata['seed']
    # model
    descriptor = {}
    descriptor['type'] = 'loc_frame'
    descriptor['sel_a'] = jdata['sel_a']
    descriptor['sel_r'] = jdata['sel_r']
    descriptor['rcut'] = jdata['rcut']
    descriptor['axis_rule'] = jdata['axis_rule']
    return descriptor

def _smth_descriptor(jdata):
    descriptor = {}
    seed = None
    if j_have (jdata, 'seed') :
        seed = jdata['seed']
    descriptor['type'] = 'se_a'
    descriptor['sel'] = jdata['sel_a']
    if j_have(jdata, 'rcut_smth') :
        descriptor['rcut_smth'] = jdata['rcut_smth']
    else :
        descriptor['rcut_smth'] = descriptor['rcut']
    descriptor['rcut'] = jdata['rcut']
    descriptor['neuron'] = j_must_have (jdata, 'filter_neuron')
    descriptor['axis_neuron'] = j_must_have_d (jdata, 'axis_neuron', ['n_axis_neuron'])
    descriptor['resnet_dt'] = False
    if j_have(jdata, 'resnet_dt') :
        descriptor['resnet_dt'] = jdata['filter_resnet_dt']
    if seed is not None:
        descriptor['seed'] = seed
    return descriptor

def _fitting_net(jdata):
    fitting_net = {}
    seed = None
    if j_have (jdata, 'seed') :
        seed = jdata['seed']
    fitting_net['neuron']= j_must_have_d (jdata, 'fitting_neuron', ['n_neuron'])
    fitting_net['resnet_dt'] = True
    if j_have(jdata, 'resnet_dt') :
        fitting_net['resnet_dt'] = jdata['resnet_dt']
    if j_have(jdata, 'fitting_resnet_dt') :
        fitting_net['resnet_dt'] = jdata['fitting_resnet_dt']    
    if seed is not None:
        fitting_net['seed'] = seed
    return fitting_net

def _learning_rate(jdata):
    # learning rate
    learning_rate = {}
    learning_rate['type'] = 'exp'
    learning_rate['decay_steps'] = j_must_have(jdata, 'decay_steps')
    learning_rate['decay_rate'] = j_must_have(jdata, 'decay_rate')
    learning_rate['start_lr'] = j_must_have(jdata, 'start_lr')
    return learning_rate

def _loss(jdata):
    # loss
    loss = {}
    loss['start_pref_e'] = j_must_have (jdata, 'start_pref_e')
    loss['limit_pref_e'] = j_must_have (jdata, 'limit_pref_e')
    loss['start_pref_f'] = j_must_have (jdata, 'start_pref_f')
    loss['limit_pref_f'] = j_must_have (jdata, 'limit_pref_f')
    loss['start_pref_v'] = j_must_have (jdata, 'start_pref_v')
    loss['limit_pref_v'] = j_must_have (jdata, 'limit_pref_v')
    if j_have(jdata, 'start_pref_ae') :
        loss['start_pref_ae'] = jdata['start_pref_ae']
    if j_have(jdata, 'limit_pref_ae') :
        loss['limit_pref_ae'] = jdata['limit_pref_ae']
    return loss

def _training(jdata):
    # training
    training = {}
    seed = None
    if j_have (jdata, 'seed') :
        seed = jdata['seed']
    training['systems'] = jdata['systems']
    training['set_prefix'] = jdata['set_prefix']
    training['stop_batch'] = jdata['stop_batch']
    training['batch_size'] = jdata['batch_size']
    if seed is not None:
        training['seed'] = seed
    training['disp_file'] = "lcurve.out"
    if j_have (jdata, "disp_file") : training['disp_file'] = jdata["disp_file"]
    training['disp_freq'] = j_must_have (jdata, 'disp_freq')
    training['numb_test'] = j_must_have (jdata, 'numb_test')
    training['save_freq'] = j_must_have (jdata, 'save_freq')
    training['save_ckpt'] = j_must_have (jdata, 'save_ckpt')
    training['disp_training'] = j_must_have (jdata, 'disp_training')
    training['time_training'] = j_must_have (jdata, 'time_training')
    if j_have (jdata, 'profiling') :
        training['profiling'] = jdata['profiling']
        if training['profiling'] :
            training['profiling_file'] = j_must_have (jdata, 'profiling_file')
    return training
