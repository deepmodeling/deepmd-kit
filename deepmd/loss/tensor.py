import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float

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
        self.atomic = kwarg.get('atomic', None)
        if jdata is not None:
            self.scale = jdata.get('scale', 1.0)
        else:
            self.scale = 1.0

        # YHT: added for global / local dipole combination
        if self.atomic is True: # upper regulation, will control the lower behavior
            self.local_weight,self.global_weight = 1.0,0.0
        elif self.atomic is False: # upper regulation, will control the lower behavior
            self.local_weight,self.global_weight = 0.0,1.0
        else: # self.atomic is None, let the loss parameter decide which mode to use
            if jdata is not None:
                self.local_weight = jdata.get('pref_atomic_' + self.tensor_name,None)
                self.global_weight = jdata.get('pref_' + self.tensor_name,None)

                # get the input parameter first
                if self.local_weight is None and self.global_weight is None:
                    # default: downward compatibility, using local mode
                    self.local_weight , self.global_weight = 1.0, 0.0
                    self.atomic = True
                elif self.local_weight is None and self.global_weight is not None:
                    # using global mode only, normalize to 1
                    assert self.global_weight > 0.0, AssertionError('assign a zero weight to global dipole without setting a local weight')
                    self.local_weight = 0.0
                    self.atomic = False
                elif self.local_weight is not None and self.global_weight is None:
                    assert self.local_weight > 0.0, AssertionError('assign a zero weight to local dipole without setting a global weight')
                    self.global_weight = 0.0
                    self.atomic = True
                else:   # Both are not None
                    self.atomic = True if self.local_weight != 0.0 else False
                    assert (self.local_weight >0.0) or (self.global_weight>0.0), AssertionError('can not assian zero weight to both local and global mode')

                    # normalize, not do according to Han Wang's suggestion
                    #temp_sum = self.local_weight + self.global_weight
                    #self.local_weight   /=  temp_sum
                    #self.global_weight  /=  temp_sum
                    
            else: # Nothing been set, use default setting
                self.local_weight,self.global_weight = 1.0,0.0
                self.atomic = True

        # data required
        add_data_requirement(self.label_name, 
                             self.tensor_size, 
                             atomic=self.atomic,  
                             must=True, 
                             high_prec=False, 
                             type_sel = self.type_sel)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        polar_hat = label_dict[self.label_name]
        polar = model_dict[self.tensor_name]
        
        # YHT: added for global / local dipole combination
        l2_loss = global_cvt_2_tf_float(0.0)
        more_loss = {
            "local_loss":global_cvt_2_tf_float(0.0),
            "global_loss":global_cvt_2_tf_float(0.0)
        }

        
        if self.local_weight > 0.0:
            local_loss = tf.reduce_mean( tf.square(self.scale*(polar - polar_hat)), name='l2_'+suffix)
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
            if self.atomic: # If label is local, however
                global_polar_hat = tf.reshape(tf.reduce_sum(tf.reshape(
                    polar_hat, [nframes, -1, self.tensor_size]), axis=1),[-1])
            else:
                global_polar_hat = polar_hat
            
            global_loss = tf.reduce_mean( tf.square(self.scale*(global_polar - global_polar_hat)), name='l2_'+suffix)

            more_loss['global_loss'] = global_loss
            self.l2_loss_global_summary = tf.summary.scalar('l2_global_loss', 
                                            tf.sqrt(more_loss['global_loss']) / global_cvt_2_tf_float(atoms))

            # YHT: should only consider atoms with dipole, i.e. atoms
            # atom_norm  = 1./ global_cvt_2_tf_float(natoms[0])  
            atom_norm  = 1./ global_cvt_2_tf_float(atoms)  
            global_loss *= atom_norm   

            l2_loss += self.global_weight * global_loss
            
        self.l2_more = more_loss
        self.l2_l = l2_loss

        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        return l2_loss, more_loss

    def print_header(self):
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
                          feed_dict_batch) :

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
        error_train = sess.run(run_data, feed_dict=feed_dict_batch)

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            #summary_merged_op = tf.summary.merge([self.l2_loss_summary])
            summary_merged_op = tf.summary.merge(summary_list)
            run_data.insert(0, summary_merged_op)

        test_out = sess.run(run_data, feed_dict=feed_dict_test)

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
