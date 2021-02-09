import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement

from deepmd.run_options import global_cvt_2_tf_float
from deepmd.run_options import global_cvt_2_ener_float

class TensorLoss () :
    """
    Loss function for tensorial properties.
    """
    def __init__ (self, jdata, **kwarg) :
        try:
            model = kwarg['model']
            type_sel = model.get_sel_type()
        except :
            type_sel = None
        self.tensor_name = kwarg['tensor_name']
        self.tensor_size = kwarg['tensor_size']
        self.label_name = kwarg['label_name']
        self.atomic = kwarg.get('atomic', True)
        if jdata is not None:
            self.scale = jdata.get('scale', 1.0)
        else:
            self.scale = 1.0
        # data required
        add_data_requirement(self.label_name, 
                             self.tensor_size, 
                             atomic=self.atomic,  
                             must=True, 
                             high_prec=False, 
                             type_sel = type_sel)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        polar_hat = label_dict[self.label_name]
        polar = model_dict[self.tensor_name]
        l2_loss = tf.reduce_mean( tf.square(self.scale*(polar - polar_hat)), name='l2_'+suffix)
        more_loss = {'nonorm': l2_loss}
        if not self.atomic :
            atom_norm  = 1./ global_cvt_2_tf_float(natoms[0]) 
            l2_loss = l2_loss * atom_norm
        self.l2_l = l2_loss
        self.l2_more = more_loss['nonorm']

        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        return l2_loss, more_loss

    @staticmethod
    def print_header():
        prop_fmt = '   %9s %9s'
        print_str = ''
        print_str += prop_fmt % ('l2_tst', 'l2_trn')
        return print_str

    def print_on_training(self, 
                          tb_writer,
                          cur_batch,
                          sess, 
                          natoms,
                          feed_dict_test,
                          feed_dict_batch) :

        run_data = [self.l2_l]

        # first train data
        error_train = sess.run(run_data, feed_dict=feed_dict_batch)

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            summary_merged_op = tf.summary.merge([self.l2_loss_summary])
            run_data.insert(0, summary_merged_op)

        test_out = sess.run(run_data, feed_dict=feed_dict_test)

        if tb_writer:
            summary = test_out.pop(0)
            tb_writer.add_summary(summary, cur_batch)

        error_test = test_out[0]     
        
        print_str = ""
        prop_fmt = "   %9.2e %9.2e"
        print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))

        return print_str
