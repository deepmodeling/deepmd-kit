import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float

class EnerStdLoss () :
    """
    Standard loss function for DP models
    """
    def __init__ (self, 
                  starter_learning_rate : float, 
                  start_pref_e : float = 0.02,
                  limit_pref_e : float = 1.00,
                  start_pref_f : float = 1000,
                  limit_pref_f : float = 1.00,
                  start_pref_v : float = 0.0,
                  limit_pref_v : float = 0.0,
                  start_pref_ae : float = 0.0,
                  limit_pref_ae : float = 0.0,
                  start_pref_pf : float = 0.0,
                  limit_pref_pf : float = 0.0,
                  relative_f : float = None 
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.relative_f = relative_f
        self.has_e = (self.start_pref_e != 0.0 or self.limit_pref_e != 0.0)
        self.has_f = (self.start_pref_f != 0.0 or self.limit_pref_f != 0.0)
        self.has_v = (self.start_pref_v != 0.0 or self.limit_pref_v != 0.0)
        self.has_ae = (self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0)
        self.has_pf = (self.start_pref_pf != 0.0 or self.limit_pref_pf != 0.0)
        # data required
        add_data_requirement('energy', 1, atomic=False, must=False, high_prec=True)
        add_data_requirement('force',  3, atomic=True,  must=False, high_prec=False)
        add_data_requirement('virial', 9, atomic=False, must=False, high_prec=False)
        add_data_requirement('atom_ener', 1, atomic=True, must=False, high_prec=False)
        add_data_requirement('atom_pref', 1, atomic=True, must=False, high_prec=False, repeat=3)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        energy = model_dict['energy']
        force = model_dict['force']
        virial = model_dict['virial']
        atom_ener = model_dict['atom_ener']
        energy_hat = label_dict['energy']
        force_hat = label_dict['force']
        virial_hat = label_dict['virial']
        atom_ener_hat = label_dict['atom_ener']
        atom_pref = label_dict['atom_pref']
        find_energy = label_dict['find_energy']
        find_force = label_dict['find_force']
        find_virial = label_dict['find_virial']
        find_atom_ener = label_dict['find_atom_ener']                
        find_atom_pref = label_dict['find_atom_pref']                

        l2_ener_loss = tf.reduce_mean( tf.square(energy - energy_hat), name='l2_'+suffix)

        force_reshape = tf.reshape (force, [-1])
        force_hat_reshape = tf.reshape (force_hat, [-1])
        atom_pref_reshape = tf.reshape (atom_pref, [-1])
        diff_f = force_hat_reshape - force_reshape
        if self.relative_f is not None:            
            force_hat_3 = tf.reshape(force_hat, [-1, 3])
            norm_f = tf.reshape(tf.norm(force_hat_3, axis = 1), [-1, 1]) + self.relative_f
            diff_f_3 = tf.reshape(diff_f, [-1, 3])
            diff_f_3 = diff_f_3 / norm_f
            diff_f = tf.reshape(diff_f_3, [-1])
        l2_force_loss = tf.reduce_mean(tf.square(diff_f), name = "l2_force_" + suffix)
        l2_pref_force_loss = tf.reduce_mean(tf.multiply(tf.square(diff_f), atom_pref_reshape), name = "l2_pref_force_" + suffix)

        virial_reshape = tf.reshape (virial, [-1])
        virial_hat_reshape = tf.reshape (virial_hat, [-1])
        l2_virial_loss = tf.reduce_mean (tf.square(virial_hat_reshape - virial_reshape), name = "l2_virial_" + suffix)

        atom_ener_reshape = tf.reshape (atom_ener, [-1])
        atom_ener_hat_reshape = tf.reshape (atom_ener_hat, [-1])
        l2_atom_ener_loss = tf.reduce_mean (tf.square(atom_ener_hat_reshape - atom_ener_reshape), name = "l2_atom_ener_" + suffix)

        atom_norm  = 1./ global_cvt_2_tf_float(natoms[0]) 
        atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms[0]) 
        pref_e = global_cvt_2_ener_float(find_energy * (self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * learning_rate / self.starter_learning_rate) )
        pref_f = global_cvt_2_tf_float(find_force * (self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * learning_rate / self.starter_learning_rate) )
        pref_v = global_cvt_2_tf_float(find_virial * (self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * learning_rate / self.starter_learning_rate) )
        pref_ae= global_cvt_2_tf_float(find_atom_ener * (self.limit_pref_ae+ (self.start_pref_ae-self.limit_pref_ae) * learning_rate / self.starter_learning_rate) )
        pref_pf= global_cvt_2_tf_float(find_atom_pref * (self.limit_pref_pf+ (self.start_pref_pf-self.limit_pref_pf) * learning_rate / self.starter_learning_rate) )

        l2_loss = 0
        more_loss = {}
        if self.has_e :
            l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
        more_loss['l2_ener_loss'] = l2_ener_loss
        if self.has_f :
            l2_loss += global_cvt_2_ener_float(pref_f * l2_force_loss)
        more_loss['l2_force_loss'] = l2_force_loss
        if self.has_v :
            l2_loss += global_cvt_2_ener_float(atom_norm * (pref_v * l2_virial_loss))
        more_loss['l2_virial_loss'] = l2_virial_loss
        if self.has_ae :
            l2_loss += global_cvt_2_ener_float(pref_ae * l2_atom_ener_loss)
        more_loss['l2_atom_ener_loss'] = l2_atom_ener_loss
        if self.has_pf :
            l2_loss += global_cvt_2_ener_float(pref_pf * l2_pref_force_loss)
        more_loss['l2_pref_force_loss'] = l2_pref_force_loss

        # only used when tensorboard was set as true
        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar('l2_ener_loss', global_cvt_2_tf_float(tf.sqrt(l2_ener_loss)) / global_cvt_2_tf_float(natoms[0]))
        self.l2_loss_force_summary = tf.summary.scalar('l2_force_loss', tf.sqrt(l2_force_loss))
        self.l2_loss_virial_summary = tf.summary.scalar('l2_virial_loss', tf.sqrt(l2_virial_loss) / global_cvt_2_tf_float(natoms[0]))

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def print_header(self):
        prop_fmt = '   %11s %11s'
        print_str = ''
        print_str += prop_fmt % ('rmse_tst', 'rmse_trn')
        if self.has_e :
            print_str += prop_fmt % ('rmse_e_tst', 'rmse_e_trn')
        if self.has_ae :
            print_str += prop_fmt % ('rmse_ae_tst', 'rmse_ae_trn')
        if self.has_f :
            print_str += prop_fmt % ('rmse_f_tst', 'rmse_f_trn')
        if self.has_v :
            print_str += prop_fmt % ('rmse_v_tst', 'rmse_v_trn')
        if self.has_pf :
            print_str += prop_fmt % ('rmse_pf_tst', 'rmse_pf_trn')
        return print_str

    def print_on_training(self, 
                          tb_writer,
                          cur_batch,
                          sess, 
                          natoms,
                          feed_dict_test,
                          feed_dict_batch):

        run_data = [
            self.l2_l,
            self.l2_more['l2_ener_loss'],
            self.l2_more['l2_force_loss'],
            self.l2_more['l2_virial_loss'],
            self.l2_more['l2_atom_ener_loss'],
            self.l2_more['l2_pref_force_loss']
        ]

        # first train data
        train_out = sess.run(run_data, feed_dict=feed_dict_batch)
        error_train, error_e_train, error_f_train, error_v_train, error_ae_train, error_pf_train = train_out

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            summary_merged_op = tf.summary.merge([self.l2_loss_summary, self.l2_loss_ener_summary, self.l2_loss_force_summary, self.l2_loss_virial_summary])
            run_data.insert(0, summary_merged_op)

        test_out = sess.run(run_data, feed_dict=feed_dict_test)

        if tb_writer:
            summary = test_out.pop(0)
            tb_writer.add_summary(summary, cur_batch)

        error_test, error_e_test, error_f_test, error_v_test, error_ae_test, error_pf_test = test_out

        
        print_str = ""
        prop_fmt = "   %11.2e %11.2e"
        print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
        if self.has_e :
            print_str += prop_fmt % (np.sqrt(error_e_test) / natoms[0], np.sqrt(error_e_train) / natoms[0])
        if self.has_ae :
            print_str += prop_fmt % (np.sqrt(error_ae_test), np.sqrt(error_ae_train))
        if self.has_f :
            print_str += prop_fmt % (np.sqrt(error_f_test), np.sqrt(error_f_train))
        if self.has_v :
            print_str += prop_fmt % (np.sqrt(error_v_test) / natoms[0], np.sqrt(error_v_train) / natoms[0])
        if self.has_pf:
            print_str += prop_fmt % (np.sqrt(error_pf_test), np.sqrt(error_pf_train))

        return print_str      


class EnerDipoleLoss () :
    def __init__ (self, 
                  starter_learning_rate : float,
                  start_pref_e : float = 0.1,
                  limit_pref_e : float = 1.0,
                  start_pref_ed : float = 1.0,
                  limit_pref_ed : float = 1.0
    ) -> None :
        self.starter_learning_rate = kwarg['starter_learning_rate']
        args = ClassArg()\
            .add('start_pref_e',        float,  must = True, default = 0.1) \
            .add('limit_pref_e',        float,  must = True, default = 1.00)\
            .add('start_pref_ed',       float,  must = True, default = 1.00)\
            .add('limit_pref_ed',       float,  must = True, default = 1.00)
        class_data = args.parse(jdata)
        self.start_pref_e = class_data['start_pref_e']
        self.limit_pref_e = class_data['limit_pref_e']
        self.start_pref_ed = class_data['start_pref_ed']
        self.limit_pref_ed = class_data['limit_pref_ed']
        # data required
        add_data_requirement('energy', 1, atomic=False, must=True, high_prec=True)
        add_data_requirement('energy_dipole', 3, atomic=False, must=True, high_prec=False)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        coord = model_dict['coord']
        energy = model_dict['energy']
        atom_ener = model_dict['atom_ener']
        nframes = tf.shape(atom_ener)[0]
        natoms = tf.shape(atom_ener)[1]
        # build energy dipole
        atom_ener0 = atom_ener - tf.reshape(tf.tile(tf.reshape(energy/global_cvt_2_ener_float(natoms), [-1, 1]), [1, natoms]), [nframes, natoms])
        coord = tf.reshape(coord, [nframes, natoms, 3])
        atom_ener0 = tf.reshape(atom_ener0, [nframes, 1, natoms])
        ener_dipole = tf.matmul(atom_ener0, coord)
        ener_dipole = tf.reshape(ener_dipole, [nframes, 3])
        
        energy_hat = label_dict['energy']
        ener_dipole_hat = label_dict['energy_dipole']
        find_energy = label_dict['find_energy']
        find_ener_dipole = label_dict['find_energy_dipole']                

        l2_ener_loss = tf.reduce_mean( tf.square(energy - energy_hat), name='l2_'+suffix)

        ener_dipole_reshape = tf.reshape(ener_dipole, [-1])
        ener_dipole_hat_reshape = tf.reshape(ener_dipole_hat, [-1])
        l2_ener_dipole_loss = tf.reduce_mean( tf.square(ener_dipole_reshape - ener_dipole_hat_reshape), name='l2_'+suffix)

        # atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms[0]) 
        atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms) 
        pref_e  = global_cvt_2_ener_float(find_energy * (self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * learning_rate / self.starter_learning_rate) )
        pref_ed = global_cvt_2_tf_float(find_ener_dipole * (self.limit_pref_ed + (self.start_pref_ed - self.limit_pref_ed) * learning_rate / self.starter_learning_rate) )

        l2_loss = 0
        more_loss = {}
        l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
        l2_loss += global_cvt_2_ener_float(pref_ed * l2_ener_dipole_loss)
        more_loss['l2_ener_loss'] = l2_ener_loss
        more_loss['l2_ener_dipole_loss'] = l2_ener_dipole_loss

        self.l2_loss_summary = tf.summary.scalar('l2_loss', tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar('l2_ener_loss', tf.sqrt(l2_ener_loss) / global_cvt_2_tf_float(natoms[0]))
        self.l2_ener_dipole_loss_summary = tf.summary.scalar('l2_ener_dipole_loss', tf.sqrt(l2_ener_dipole_loss))

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    @staticmethod
    def print_header() :
        prop_fmt = '   %9s %9s'
        print_str = ''
        print_str += prop_fmt % ('l2_tst', 'l2_trn')
        print_str += prop_fmt % ('l2_e_tst', 'l2_e_trn')
        print_str += prop_fmt % ('l2_ed_tst', 'l2_ed_trn')
        return print_str 

    def print_on_training(self,
                          tb_writer,
                          cur_batch, 
                          sess, 
                          natoms,
                          feed_dict_test,
                          feed_dict_batch):

        run_data = [
            self.l2_l,
            self.l2_more['l2_ener_loss'],
            self.l2_more['l2_ener_dipole_loss']
        ]

        # first train data
        train_out = sess.run(run_data, feed_dict=feed_dict_batch)
        error_train, error_e_train, error_ed_train = train_out

        # than test data, if tensorboard log writter is present, commpute summary
        # and write tensorboard logs
        if tb_writer:
            summary_merged_op = tf.summary.merge([
                self.l2_loss_summary,
                self.l2_loss_ener_summary,
                self.l2_ener_dipole_loss_summary
            ])
            run_data.insert(0, summary_merged_op)

        test_out = sess.run(run_data, feed_dict=feed_dict_test)

        if tb_writer:
            summary = test_out.pop(0)
            tb_writer.add_summary(summary, cur_batch)

        error_test, error_e_test, error_ed_test = test_out

        print_str = ""
        prop_fmt = "   %9.2e %9.2e"
        print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
        print_str += prop_fmt % (np.sqrt(error_e_test) / natoms[0], np.sqrt(error_e_train) / natoms[0])
        print_str += prop_fmt % (np.sqrt(error_ed_test), np.sqrt(error_ed_train))
        return print_str   



