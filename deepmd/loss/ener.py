import numpy as np
from deepmd.env import tf
from deepmd.common import add_data_requirement

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import global_cvt_2_ener_float
from deepmd.utils.sess import run_sess
from .loss import Loss


class EnerStdLoss (Loss) :
    """
    Standard loss function for DP models

    Parameters
    ----------
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \\sum_i c_i E_i
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
                  relative_f : float = None,
                  enable_atom_ener_coeff: bool=False,
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
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
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
        if self.enable_atom_ener_coeff:
            add_data_requirement('atom_ener_coeff', 1, atomic=True, must=False, high_prec=False, default=1.)

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

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as 
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener_coeff = label_dict['atom_ener_coeff']
            atom_ener_coeff = tf.reshape(atom_ener_coeff, tf.shape(atom_ener))
            energy = tf.reduce_sum(atom_ener_coeff * atom_ener, 1)
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
        self.l2_loss_summary = tf.summary.scalar('l2_loss_' + suffix, tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar('l2_ener_loss_' + suffix, global_cvt_2_tf_float(tf.sqrt(l2_ener_loss)) / global_cvt_2_tf_float(natoms[0]))
        self.l2_loss_force_summary = tf.summary.scalar('l2_force_loss_' + suffix, tf.sqrt(l2_force_loss))
        self.l2_loss_virial_summary = tf.summary.scalar('l2_virial_loss_' + suffix, tf.sqrt(l2_virial_loss) / global_cvt_2_tf_float(natoms[0]))

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more['l2_ener_loss'] if self.has_e else placeholder,
            self.l2_more['l2_force_loss'] if self.has_f else placeholder,
            self.l2_more['l2_virial_loss'] if self.has_v else placeholder,
            self.l2_more['l2_atom_ener_loss'] if self.has_ae else placeholder,
            self.l2_more['l2_pref_force_loss'] if self.has_pf else placeholder,
        ]
        error, error_e, error_f, error_v, error_ae, error_pf = run_sess(sess, run_data, feed_dict=feed_dict)
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        if self.has_e:
            results["rmse_e"] = np.sqrt(error_e) / natoms[0]
        if self.has_ae:
            results["rmse_ae"] = np.sqrt(error_ae)
        if self.has_f:
            results["rmse_f"] = np.sqrt(error_f)
        if self.has_v:
            results["rmse_v"] = np.sqrt(error_v) / natoms[0]
        if self.has_pf:
            results["rmse_pf"] = np.sqrt(error_pf)
        return results 


class EnerDipoleLoss (Loss) :
    def __init__ (self, 
                  starter_learning_rate : float,
                  start_pref_e : float = 0.1,
                  limit_pref_e : float = 1.0,
                  start_pref_ed : float = 1.0,
                  limit_pref_ed : float = 1.0
    ) -> None :
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_ed = start_pref_ed
        self.limit_pref_ed = limit_pref_ed
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

        self.l2_loss_summary = tf.summary.scalar('l2_loss_' + suffix, tf.sqrt(l2_loss))
        self.l2_loss_ener_summary = tf.summary.scalar('l2_ener_loss_' + suffix, tf.sqrt(l2_ener_loss) / global_cvt_2_tf_float(natoms[0]))
        self.l2_ener_dipole_loss_summary = tf.summary.scalar('l2_ener_dipole_loss_' + suffix, tf.sqrt(l2_ener_dipole_loss))

        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        run_data = [
            self.l2_l,
            self.l2_more['l2_ener_loss'],
            self.l2_more['l2_ener_dipole_loss']
        ]
        error, error_e, error_ed = run_sess(sess, run_data, feed_dict=feed_dict)
        results = {
            'natoms': natoms[0],
            'rmse': np.sqrt(error),
            'rmse_e': np.sqrt(error_e) / natoms[0],
            'rmse_ed': np.sqrt(error_ed)
        }
        return results
