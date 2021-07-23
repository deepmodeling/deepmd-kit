import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float
from deepmd.utils.sess import run_sess

class TensorLoss () :
    """
    Loss function for tensorial properties.
    """
    def __init__ (self, jdata, **kwarg) :
        try:
            model = kwarg['model']
            self.type_sel = model.get_sel_type()
        except :
            self.type_sel = None
        self.tensor_name = kwarg['tensor_name']
        self.tensor_size = kwarg['tensor_size']
        self.label_name = kwarg['label_name']
        if jdata is not None:
            self.scale = jdata.get('scale', 1.0)
        else:
            self.scale = 1.0

        # YHT: added for global / local dipole combination
        assert jdata is not None, "Please provide loss parameters!"
        # YWolfeee: modify, use pref / pref_atomic, instead of pref_weight / pref_atomic_weight
        self.local_weight = jdata.get('pref_atomic', None)
        self.global_weight = jdata.get('pref', None)

        assert (self.local_weight is not None and self.global_weight is not None), "Both `pref` and `pref_atomic` should be provided."
        assert self.local_weight >= 0.0 and self.global_weight >= 0.0, "Can not assign negative weight to `pref` and `pref_atomic`"
        assert (self.local_weight >0.0) or (self.global_weight>0.0), AssertionError('Can not assian zero weight both to `pref` and `pref_atomic`')

        # data required
        add_data_requirement("atomic_" + self.label_name, 
                             self.tensor_size, 
                             atomic=True,  
                             must=False, 
                             high_prec=False, 
                             type_sel = self.type_sel)
        add_data_requirement(self.label_name, 
                             self.tensor_size, 
                             atomic=False,  
                             must=False, 
                             high_prec=False, 
                             type_sel = self.type_sel)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        polar_hat = label_dict[self.label_name]
        atomic_polar_hat = label_dict["atomic_" + self.label_name]
        polar = tf.reshape(model_dict[self.tensor_name], [-1])

        find_global = label_dict['find_' + self.label_name]
        find_atomic = label_dict['find_atomic_' + self.label_name]
        
        

        # YHT: added for global / local dipole combination
        l2_loss = global_cvt_2_tf_float(0.0)
        more_loss = {
            "local_loss":global_cvt_2_tf_float(0.0),
            "global_loss":global_cvt_2_tf_float(0.0)
        }

        
        if self.local_weight > 0.0:
            local_loss = global_cvt_2_tf_float(find_atomic) * tf.reduce_mean( tf.square(self.scale*(polar - atomic_polar_hat)), name='l2_'+suffix)
            more_loss['local_loss'] = local_loss
            l2_loss += self.local_weight * local_loss
            self.l2_loss_local_summary = tf.summary.scalar('l2_local_loss', 
                                            tf.sqrt(more_loss['local_loss']))
        

        if self.global_weight > 0.0:    # Need global loss
            atoms = 0
            if self.type_sel is not None:
                for w in self.type_sel:
                    atoms += natoms[2+w]
            else:
                atoms = natoms[0]     
            nframes = tf.shape(polar)[0] // self.tensor_size // atoms
            # get global results
            global_polar = tf.reshape(tf.reduce_sum(tf.reshape(
                polar, [nframes, -1, self.tensor_size]), axis=1),[-1])
            #if self.atomic: # If label is local, however
            #    global_polar_hat = tf.reshape(tf.reduce_sum(tf.reshape(
            #        polar_hat, [nframes, -1, self.tensor_size]), axis=1),[-1])
            #else:
            #    global_polar_hat = polar_hat
            
            global_loss = global_cvt_2_tf_float(find_global) * tf.reduce_mean( tf.square(self.scale*(global_polar - polar_hat)), name='l2_'+suffix)

            more_loss['global_loss'] = global_loss
            self.l2_loss_global_summary = tf.summary.scalar('l2_global_loss', 
                                            tf.sqrt(more_loss['global_loss']) / global_cvt_2_tf_float(atoms))

            # YWolfeee: should only consider atoms with dipole, i.e. atoms
            # atom_norm  = 1./ global_cvt_2_tf_float(natoms[0])  
            atom_norm  = 1./ global_cvt_2_tf_float(atoms)  
            global_loss *= atom_norm   

            l2_loss += self.global_weight * global_loss
            
        self.l2_more = more_loss
        self.l2_l = l2_loss

        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        atoms = 0
        if self.type_sel is not None:
            for w in self.type_sel:
                atoms += natoms[2+w]
        else:
            atoms = natoms[0]

        run_data = [self.l2_l, self.l2_more['local_loss'], self.l2_more['global_loss']]
        error, error_lc, error_gl = run_sess(sess, run_data, feed_dict=feed_dict)

        results = {"natoms": atoms, "rmse": np.sqrt(error)}
        if self.local_weight > 0.0:
            results["rmse_lc"] = np.sqrt(error_lc)
        if self.global_weight > 0.0:
            results["rmse_gl"] = np.sqrt(error_gl) / atoms
        return results

    def print_header(self):  # depreciated
        prop_fmt = '   %11s %11s'
        print_str = ''
        print_str += prop_fmt % ('rmse_tst', 'rmse_trn')
        if self.local_weight > 0.0:
            print_str += prop_fmt % ('rmse_lc_tst', 'rmse_lc_trn')
        if self.global_weight > 0.0:
            print_str += prop_fmt % ('rmse_gl_tst', 'rmse_gl_trn')
        return print_str

    def print_on_training(self, 
                          tb_writer,
                          cur_batch,
                          sess, 
                          natoms,
                          feed_dict_test,
                          feed_dict_batch) :  # depreciated

        # YHT: added to calculate the atoms number
        atoms = 0
        if self.type_sel is not None:
            for w in self.type_sel:
                atoms += natoms[2+w]                   
        else:
            atoms = natoms[0]

        run_data = [self.l2_l, self.l2_more['local_loss'], self.l2_more['global_loss']]
        summary_list = [self.l2_loss_summary]
        if self.local_weight > 0.0:
            summary_list.append(self.l2_loss_local_summary)
        if self.global_weight > 0.0:
            summary_list.append(self.l2_loss_global_summary)

        # first train data
        error_train = run_sess(sess, run_data, feed_dict=feed_dict_batch)

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            #summary_merged_op = tf.summary.merge([self.l2_loss_summary])
            summary_merged_op = tf.summary.merge(summary_list)
            run_data.insert(0, summary_merged_op)

        test_out = run_sess(sess, run_data, feed_dict=feed_dict_test)

        if tb_writer:
            summary = test_out.pop(0)
            tb_writer.add_summary(summary, cur_batch)

        error_test = test_out  
        
        print_str = ""
        prop_fmt = "   %11.2e %11.2e"
        print_str += prop_fmt % (np.sqrt(error_test[0]), np.sqrt(error_train[0]))
        if self.local_weight > 0.0:
            print_str += prop_fmt % (np.sqrt(error_test[1]), np.sqrt(error_train[1]) )
        if self.global_weight > 0.0:
            print_str += prop_fmt % (np.sqrt(error_test[2])/atoms, np.sqrt(error_train[2])/atoms)

        return print_str
