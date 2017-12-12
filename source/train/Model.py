#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

# load force module
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.so" )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.so")

# load grad of force module
sys.path.append (module_path )
import deepmd._prod_force_grad
import deepmd._prod_virial_grad

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json data base must provide key " + key )
    else :
        return jdata[key]

def j_have (jdata, key) :
    return key in jdata.keys() 

class LearingRate (object) :
    def __init__ (self, 
                  jdata, 
                  tot_numb_batches) :
        self.decay_steps_ = j_must_have(jdata, 'decay_steps')
        self.decay_rate_ = j_must_have(jdata, 'decay_rate')
        self.start_lr_ = j_must_have(jdata, 'start_lr')        
        self.tot_numb_batches = tot_numb_batches

    def value (self, 
              batch) :
        return self.start_lr_ * np.power (self.decay_rate_, (batch // self.decay_steps()))

    def decay_steps (self) :
#        return self.decay_steps_ * self.tot_numb_batches
        return self.decay_steps_
    
    def decay_rate (self) : 
        return self.decay_rate_

    def start_lr (self) :
        return self.start_lr_

class NNPModel (object):
    def __init__(self, 
                 jdata, 
                 sess):
        self.sess = sess
        # descrpt config
        self.sel_a = j_must_have (jdata, 'sel_a')
        self.sel_r = j_must_have (jdata, 'sel_r')
        self.rcut_a = j_must_have (jdata, 'rcut_a')
        self.rcut_r = j_must_have (jdata, 'rcut_r')
        self.axis_rule = []
        if j_have (jdata, 'axis_rule') :
            self.axis_rule = jdata['axis_rule']                    
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        # copy from config
        self.n_neuron = j_must_have (jdata, 'n_neuron')

        self.batch_size = j_must_have (jdata, 'batch_size')
        self.numb_test = j_must_have (jdata, 'numb_test')
        self.useBN = False
        if j_have (jdata, 'useBN') :
            self.useBN = jdata['useBN']

        self.start_pref_e = j_must_have (jdata, 'start_pref_e')
        self.limit_pref_e = j_must_have (jdata, 'limit_pref_e')
        self.start_pref_f = j_must_have (jdata, 'start_pref_f')
        self.limit_pref_f = j_must_have (jdata, 'limit_pref_f')
        self.start_pref_v = j_must_have (jdata, 'start_pref_v')
        self.limit_pref_v = j_must_have (jdata, 'limit_pref_v')
        self.has_e = (self.start_pref_e != 0 or self.limit_pref_e != 0)
        self.has_f = (self.start_pref_f != 0 or self.limit_pref_f != 0)
        self.has_v = (self.start_pref_v != 0 or self.limit_pref_v != 0)

        self.disp_file = "lcurve.out"
        if j_have (jdata, "disp_file") : self.disp_file = jdata["disp_file"]
        self.disp_freq = j_must_have (jdata, 'disp_freq')
        self.save_freq = j_must_have (jdata, 'save_freq')
        self.save_ckpt = j_must_have (jdata, 'save_ckpt')
        self.restart = j_must_have (jdata, 'restart')
        self.load_model = os.getcwd() + "/" + j_must_have (jdata, 'load_ckpt')

        self.seed = None
        if j_have (jdata, 'seed') :
            self.seed = jdata['seed']
        self.num_threads = j_must_have (jdata, 'num_threads')

        self.display_in_training = j_must_have (jdata, 'disp_training')
        self.timing_in_training = j_must_have (jdata, 'time_training')

        self.null_mesh = tf.constant ([-1])        

    def build (self, 
               data, 
               lr) :
        self.ntypes = len(self.sel_a)
        assert (self.ntypes == len(self.sel_r)), "size sel r array should match ntypes"

        natoms_vec = data.get_natoms_vec (self.ntypes)
        natoms_vec = natoms_vec.astype(np.int32)
        self.ncopies = np.cumprod(data.get_ncopies ())[-1]

        test_prop_c, test_energy, test_force, test_virial, test_coord, test_box, test_type = data.get_test ()

        ncell = np.ones (3, dtype=np.int32)
        cell_size = np.max ([self.rcut_r, self.rcut_a])
        avg_box = np.average (test_box, axis = 0)
        avg_box = np.reshape (avg_box, [3,3])
        for ii in range (3) :
            ncell[ii] = int ( np.linalg.norm(avg_box[ii]) / cell_size )
            if (ncell[ii] < 2) : ncell[ii] = 2
        default_mesh = np.zeros (6, dtype = np.int32)
        default_mesh[3] = ncell[0]
        default_mesh[4] = ncell[1]
        default_mesh[5] = ncell[2]
        t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), name = 't_rcut', dtype = tf.float64)

        test_coord_ = test_coord[:self.numb_test,:]
        test_box_ = test_box[:self.numb_test,:]
        test_type_ = test_type[:self.numb_test,:]

        self.compute_stats (test_coord_, test_box_, test_type_, natoms_vec, default_mesh)
        print ("# computed test stats")
        sys.stdout.flush()

        bias_atom_e = data.get_bias_atom_e()

        self.t_prop_c           = tf.placeholder(tf.float32, [3],    name='t_prop_c')
        self.t_energy           = tf.placeholder(tf.float64, [None], name='t_energy')
        self.t_force            = tf.placeholder(tf.float64, [None], name='t_force')
        self.t_virial           = tf.placeholder(tf.float64, [None], name='t_virial')
        self.t_coord            = tf.placeholder(tf.float64, [None], name='t_coord')
        self.t_type             = tf.placeholder(tf.int32,   [None], name='t_type')
        self.t_natoms           = tf.placeholder(tf.int32,   [self.ntypes+2], name='t_natoms')
        self.t_box              = tf.placeholder(tf.float64, [None, 9], name='t_box')
        self.t_mesh             = tf.placeholder(tf.int32,   [None], name='t_mesh')
        self.is_training        = tf.placeholder(tf.bool)

        self._extra_train_ops   = []
        self.global_step        = tf.get_variable('global_step', 
                                                  [],
                                                  initializer=tf.constant_initializer(0),
                                                  trainable=False, 
                                                  dtype=tf.int32)
        self.starter_learning_rate = lr.start_lr()
        self.learning_rate = tf.train.exponential_decay(lr.start_lr(), 
                                                        self.global_step,
                                                        lr.decay_steps(),
                                                        lr.decay_rate(), 
                                                        staircase=True)
        
        self.energy, self.force, self.virial \
            = self.build_interaction (self.t_coord, self.t_type, self.t_natoms, self.t_box, self.t_mesh, bias_atom_e = bias_atom_e, suffix = "test", reuse = False)

        self.l2_l, self.l2_el, self.l2_fl, self.l2_vl \
            = self.loss (self.ncopies, self.t_natoms, self.t_prop_c, self.t_energy, self.energy, self.t_force, self.force, self.t_virial, self.virial, suffix = "test")

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.l2_l, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        apply_op = optimizer.apply_gradients (zip (grads, trainable_variables),
                                              global_step=self.global_step,
                                              name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        saver = tf.train.Saver()

        # initialization
        if not self.restart :
            # init by init op
            print ("# initialize model")
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
        else :
            print ("# restart from model %s" % self.load_model)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            saver.restore (self.sess, self.load_model)

        fp = open(self.disp_file, "w")
        fp.close ()

    def train (self, 
               data, 
               stop_batch) :

        natoms_vec = data.get_natoms_vec (self.ntypes)
        natoms_vec = natoms_vec.astype(np.int32)

        test_prop_c, test_energy, test_force, test_virial, test_coord, test_box, test_type = data.get_test ()

        ncell = np.ones (3, dtype=np.int32)
        cell_size = np.max ([self.rcut_r, self.rcut_a])
        avg_box = np.average (test_box, axis = 0)
        avg_box = np.reshape (avg_box, [3,3])
        for ii in range (3) :
            ncell[ii] = int ( np.linalg.norm(avg_box[ii]) / cell_size )
            if (ncell[ii] < 2) : ncell[ii] = 2
        default_mesh = np.zeros (6, dtype = np.int32)
        default_mesh[3] = ncell[0]
        default_mesh[4] = ncell[1]
        default_mesh[5] = ncell[2]

        # natoms_t  :   [0] : n_loc
        #               [1] : n_all
        #             [2: ] : number of atoms for each type in the local region
        # mesh_t :    [0:3] : nat_stt
        #             [3:6] : nat_end
        #             [6:9] : ext_stt
        #             [9:12]: ext_end
        feed_dict_test = {self.t_prop_c:        test_prop_c,
                          self.t_energy:        test_energy             [:self.numb_test],
                          self.t_force:         np.reshape(test_force   [:self.numb_test, :], [-1]),
                          self.t_virial:        np.reshape(test_virial  [:self.numb_test, :], [-1]),
                          self.t_coord:         np.reshape(test_coord   [:self.numb_test, :], [-1]),
                          self.t_box:           test_box                [:self.numb_test, :],
                          self.t_type:          np.reshape(test_type    [:self.numb_test, :], [-1]),
                          self.t_natoms:        natoms_vec,
                          self.t_mesh:          default_mesh,
                          self.is_training:     False}

        # get saver
        saver = tf.train.Saver()

        fp = open(self.disp_file, "a")

        train_time = 0
        cur_batch = self.sess.run(self.global_step)
        while cur_batch < stop_batch :
            batch_prop_c, batch_energy, batch_force, batch_virial, batch_coord, batch_box, batch_type = data.get_batch (self.batch_size)
            feed_dict_batch = {self.t_prop_c:        batch_prop_c,
                               self.t_energy:        batch_energy, 
                               self.t_force:         np.reshape(batch_force, [-1]),
                               self.t_virial:        np.reshape(batch_virial, [-1]),
                               self.t_coord:         np.reshape(batch_coord, [-1]),
                               self.t_box:           batch_box,
                               self.t_type:          np.reshape(batch_type, [-1]),
                               self.t_natoms:        natoms_vec,
                               self.t_mesh:          default_mesh,
                               self.is_training:     True}
            if self.display_in_training and cur_batch == 0 :
                self.test_on_the_fly(fp, feed_dict_batch, feed_dict_test)
            if self.timing_in_training : tic = time.time()
            self.sess.run([self.train_op], feed_dict = feed_dict_batch)
            if self.timing_in_training : toc = time.time()
            if self.timing_in_training : train_time += toc - tic
            cur_batch = self.sess.run(self.global_step)

            if self.display_in_training and (cur_batch % self.disp_freq == 0) :
                tic = time.time()
                self.test_on_the_fly(fp, feed_dict_batch, feed_dict_test)
                toc = time.time()
                test_time = toc - tic
                if self.timing_in_training :
                    print ("# batch %7d training time %.2f s, testing time %.2f s" % (cur_batch, train_time, test_time))
                    train_time = 0
                if self.save_freq > 0 and cur_batch % self.save_freq == 0 :
                    save_path = saver.save (self.sess, os.getcwd() + "/" + self.save_ckpt)
                    print ("# saved checkpoint %s" % save_path)
                sys.stdout.flush()

        fp.close ()

    def get_global_step (self) :
        return self.sess.run(self.global_step)

    def print_head (self) :
        fp = open(self.disp_file, "a")
        fp.write ("# %5s   %9s %9s   %9s %9s   %9s %9s   %9s %9s   %13s\n" % ("batch", "l2_tst", "l2_trn", "l2_e_tst", "l2_e_trn", "l2_f_tst", "l2_f_trn", "l2_v_tst", "l2_v_trn", "lr"))
        fp.close ()

    def test_on_the_fly (self,
                         fp,
                         feed_dict_batch,
                         feed_dict_test) :
        error_test, error_e_test, error_f_test, error_v_test \
            = self.sess.run([self.l2_l, self.l2_el, self.l2_fl, self.l2_vl], 
                            feed_dict=feed_dict_test)
        error_train, error_e_train, error_f_train, error_v_train \
            = self.sess.run([self.l2_l, self.l2_el, self.l2_fl, self.l2_vl], 
                            feed_dict=feed_dict_batch)
        cur_batch = self.sess.run(self.global_step)
        current_lr = self.sess.run(tf.to_double(self.learning_rate))
        fp.write ("%7d   %9.2e %9.2e   %9.2e %9.2e   %9.2e %9.2e   %9.2e %9.2e   %13.6e\n" %
                  (cur_batch, 
                   np.sqrt(error_test), np.sqrt(error_train), 
                   np.sqrt(error_e_test), np.sqrt(error_e_train), 
                   np.sqrt(error_f_test), np.sqrt(error_f_train), 
                   np.sqrt(error_v_test), np.sqrt(error_v_train),
                   current_lr))
        fp.flush ()

    def compute_stats (self, 
                       data_coord, 
                       data_box, 
                       data_atype, 
                       natoms_vec,
                       mesh,
                       reuse = None) :    
        avg_zero = np.zeros(self.ndescrpt).astype(np.float64)
        std_ones = np.ones (self.ndescrpt).astype(np.float64)
        descrpt, descrpt_deriv, rij, nlist, axis \
            = op_module.descrpt (tf.constant(data_coord),
                                 tf.constant(data_atype),
                                 tf.constant(natoms_vec, dtype = tf.int32),
                                 tf.constant(data_box),
                                 tf.constant(mesh),
                                 tf.constant(avg_zero),
                                 tf.constant(std_ones),
                                 rcut_a = self.rcut_a,
                                 rcut_r = self.rcut_r,
                                 sel_a = self.sel_a,
                                 sel_r = self.sel_r,
                                 axis_rule = self.axis_rule,
                                 num_threads = self.num_threads)
        # self.sess.run(tf.global_variables_initializer())
        dd = self.sess.run (descrpt)
        dd = np.reshape (dd, [-1, self.ndescrpt])
        # print (dd.shape)
        davg = np.average (dd, axis = 0)
        dstd = np.std     (dd, axis = 0)
        # np.savetxt ("tmp.avg.out", davg)
        # np.savetxt ("tmp.std.out", dstd)
        for ii in range (len(dstd)) :
            if (np.abs(dstd[ii]) < 1e-2) :
                dstd[ii] = 1e-2
        np.savetxt ("stat.avg.out", davg)
        np.savetxt ("stat.std.out", dstd)        
        self.t_avg = tf.constant(davg.astype(np.float64))
        self.t_std = tf.constant(dstd.astype(np.float64))

    def loss (self, 
              ncopies,
              natoms,
              prop_c,
              energy, 
              energy_hat,
              force,
              force_hat, 
              virial,
              virial_hat, 
              suffix):
        l2_ener_loss = tf.reduce_mean( tf.square(energy - energy_hat), name='l2_'+suffix)

        force_reshape = tf.reshape (force, [-1])
        force_hat_reshape = tf.reshape (force_hat, [-1])

        l2_force_loss = tf.reduce_mean (tf.square(force_hat_reshape - force_reshape), name = "l2_force_" + suffix)

        virial_reshape = tf.reshape (virial, [-1])
        virial_hat_reshape = tf.reshape (virial_hat, [-1])
        l2_virial_loss = tf.reduce_mean (tf.square(virial_hat_reshape - virial_reshape), name = "l2_virial_" + suffix)

        atom_norm  = 1./ tf.to_double(natoms[0]) 
        atom_norm *= 1./ tf.to_double(ncopies * ncopies)
        pref_e = tf.to_double(prop_c[0] * (self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * self.learning_rate / self.starter_learning_rate) )
        pref_f = tf.to_double(prop_c[1] * (self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * self.learning_rate / self.starter_learning_rate) )
        pref_v = tf.to_double(prop_c[2] * (self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * self.learning_rate / self.starter_learning_rate) )

        l2_loss = 0
        if self.has_e :
            l2_loss += atom_norm * (pref_e * l2_ener_loss)
        if self.has_f :
            l2_loss += pref_f * l2_force_loss
        if self.has_v :
            l2_loss += atom_norm * (pref_v * l2_virial_loss)

        return l2_loss, l2_ener_loss, l2_force_loss, l2_virial_loss

    def build_interaction (self, 
                           coord_, 
                           atype_,
                           natoms,
                           box, 
                           mesh,
                           suffix, 
                           bias_atom_e = 0.0,
                           reuse = None):        
        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        descrpt, descrpt_deriv, rij, nlist, axis \
            = op_module.descrpt (coord,
                                 atype,
                                 natoms,
                                 box,                                    
                                 mesh,
                                 self.t_avg,
                                 self.t_std,
                                 rcut_a = self.rcut_a,
                                 rcut_r = self.rcut_r,
                                 sel_a = self.sel_a,
                                 sel_r = self.sel_r,
                                 axis_rule = self.axis_rule,
                                 num_threads = self.num_threads)

        descrpt_reshape = tf.reshape(descrpt, [-1, self.ndescrpt])
        
        atom_ener = self.build_atom_net (descrpt_reshape, natoms, bias_atom_e = bias_atom_e, reuse = reuse)

        energy_raw = tf.reshape(atom_ener, [-1, natoms[0]])
        energy = tf.reduce_sum(energy_raw, axis=1, name='energy_'+suffix)        

        net_deriv_tmp = tf.gradients (atom_ener, descrpt_reshape)
        net_deriv = net_deriv_tmp[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])

        force = op_module.prod_force (net_deriv_reshape,
                                      descrpt_deriv,
                                      nlist,
                                      axis,
                                      natoms,
                                      n_a_sel = self.nnei_a,
                                      n_r_sel = self.nnei_r,
                                      num_threads = self.num_threads)
        force = tf.reshape (force, [-1, 3 * natoms[0]], name = "force_"+suffix)
        
        virial = op_module.prod_virial (net_deriv_reshape,
                                        descrpt_deriv,
                                        rij,
                                        nlist,
                                        axis,
                                        natoms,
                                        n_a_sel = self.nnei_a,
                                        n_r_sel = self.nnei_r,
                                        num_threads = self.num_threads)
        virial = tf.reshape (virial, [-1, 9], name = "virial_"+suffix)

        return energy, force, virial
    
    def build_atom_net (self, 
                        inputs, 
                        natoms,
                        bias_atom_e = 0.0,
                        reuse = None) :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        shape = inputs.get_shape().as_list()        

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.ndescrpt],
                                 [-1, natoms[2+type_i]* self.ndescrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            start_index += natoms[2+type_i]

            # compute atom energy
            layer = self._one_layer(inputs_i, self.n_neuron[0], name='layer_0_type_'+str(type_i), reuse=reuse, seed = self.seed)
            for ii in range(1,len(self.n_neuron)) :
                layer = self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), reuse=reuse, seed = self.seed)
            final_layer = self._one_layer(layer, 1, activation_fn = None, bavg = bias_atom_e, name='final_layer_type_'+str(type_i), reuse=reuse, seed = self.seed)
            final_layer = tf.reshape(final_layer, [-1, natoms[2+type_i]])
            # final_layer = tf.cond (tf.equal(natoms[2+type_i], 0), lambda: tf.zeros((0, 0), dtype=tf.float64), lambda : tf.reshape(final_layer, [-1, natoms[2+type_i]]))

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)


        return tf.reshape(outs, [-1])

    def _one_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None):
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                tf.float64,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed))
            b = tf.get_variable('bias', 
                                [outputs_size], 
                                tf.float64,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
            hidden = tf.matmul(inputs, w) + b

        if activation_fn != None:
            if self.useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                return activation_fn(hidden)
        else:
            if self.useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
    
