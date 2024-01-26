# SPDX-License-Identifier: LGPL-3.0-or-later
import collections
import json
import unittest

import numpy as np
import tensorflow.compat.v1 as tf
import torch

tf.disable_eager_execution()

from pathlib import (
    Path,
)

from deepmd.pt.loss import (
    EnergyStdLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.learning_rate import LearningRateExp as MyLRExp
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.tf.common import (
    data_requirement,
    expand_sys_str,
)
from deepmd.tf.descriptor import DescrptSeA as DescrptSeA_tf
from deepmd.tf.fit import (
    EnerFitting,
)
from deepmd.tf.loss import (
    EnerStdLoss,
)
from deepmd.tf.model import (
    EnerModel,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.learning_rate import (
    LearningRateExp,
)

VariableState = collections.namedtuple("VariableState", ["value", "gradient"])


def torch2tf(torch_name):
    fields = torch_name.split(".")
    offset = int(fields[2] == "networks")
    element_id = int(fields[2 + offset])
    if fields[0] == "descriptor":
        layer_id = int(fields[4 + offset]) + 1
        weight_type = fields[5 + offset]
        return "filter_type_all/%s_%d_%d:0" % (weight_type, layer_id, element_id)
    elif fields[3] == "deep_layers":
        layer_id = int(fields[4])
        weight_type = fields[5]
        return "layer_%d_type_%d/%s:0" % (layer_id, element_id, weight_type)
    elif fields[3] == "final_layer":
        weight_type = fields[4]
        return "final_layer_type_%d/%s:0" % (element_id, weight_type)
    else:
        raise RuntimeError("Unexpected parameter name: %s" % torch_name)


class DpTrainer:
    def __init__(self):
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            content = fin.read()
        config = json.loads(content)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        model_config = config["model"]
        self.rcut = model_config["descriptor"]["rcut"]
        self.rcut_smth = model_config["descriptor"]["rcut_smth"]
        self.sel = model_config["descriptor"]["sel"]
        self.systems = config["training"]["validation_data"]["systems"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)
        self.batch_size = config["training"]["training_data"]["batch_size"]
        self.type_map = model_config["type_map"]
        self.filter_neuron = model_config["descriptor"]["neuron"]
        self.axis_neuron = model_config["descriptor"]["axis_neuron"]
        self.n_neuron = model_config["fitting_net"]["neuron"]
        self.data_stat_nbatch = 3
        self.start_lr = 0.001
        self.stop_lr = 3.51e-8
        self.decay_steps = 500
        self.stop_steps = 1600
        self.start_pref_e = 1.0
        self.limit_pref_e = 2.0
        self.start_pref_f = 2.0
        self.limit_pref_f = 1.0
        self.ntypes = len(self.type_map)

    def get_intermediate_state(self, num_steps=1):
        dp_model = self._get_dp_model()
        dp_loss = self._get_dp_loss()
        dp_lr = self._get_dp_lr()
        dp_ds = self._get_dp_dataset()
        dp_model.data_stat(dp_ds)

        # Build graph
        g = tf.Graph()
        with g.as_default():
            place_holders = self._get_dp_placeholders(dp_ds)
            model_pred = dp_model.build(
                coord_=place_holders["coord"],
                atype_=place_holders["type"],
                natoms=place_holders["natoms_vec"],
                box=place_holders["box"],
                mesh=place_holders["default_mesh"],
                input_dict=place_holders,
            )
            global_step = tf.train.get_or_create_global_step()
            learning_rate = dp_lr.build(global_step, self.stop_steps)
            l2_l, _ = dp_loss.build(
                learning_rate=learning_rate,
                natoms=place_holders["natoms_vec"],
                model_dict=model_pred,
                label_dict=place_holders,
                suffix="test",
            )
            t_vars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            t_grad_and_vars = optimizer.compute_gradients(l2_l, t_vars)
            train_op = optimizer.apply_gradients(t_grad_and_vars, global_step)
            init_op = tf.global_variables_initializer()
            t_heads = {
                "loss": l2_l,
                "energy": model_pred["energy"],
                "force": model_pred["force"],
                "virial": model_pred["virial"],
                "atomic_virial": model_pred["atom_virial"],
            }

        # Get statistics of each component
        stat_dict = {
            "descriptor.mean": dp_model.descrpt.davg,
            "descriptor.stddev": dp_model.descrpt.dstd,
            "fitting_net.bias_atom_e": dp_model.fitting.bias_atom_e,
        }

        # Get variables and their gradients
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            for _ in range(num_steps):
                batch = dp_ds.get_batch()
                feeds = self._get_feed_dict(batch, place_holders)
                sess.run(train_op, feed_dict=feeds)

            batch = dp_ds.get_batch()
            feeds = self._get_feed_dict(batch, place_holders)
            grads_and_vars, head_dict = sess.run(
                [t_grad_and_vars, t_heads], feed_dict=feeds
            )
            vs_dict = {}
            for idx, one in enumerate(t_vars):
                grad, var = grads_and_vars[idx]
                vs_dict[one.name] = VariableState(var, grad)

        # Used for reproducing
        return batch, head_dict, stat_dict, vs_dict

    def _get_dp_dataset(self):
        data = DeepmdDataSystem(
            systems=self.systems,
            batch_size=self.batch_size,
            test_size=1,
            rcut=self.rcut,
            type_map=self.type_map,
            trn_all_set=True,
        )
        data.add_dict(data_requirement)
        return data

    def _get_dp_model(self):
        dp_descrpt = DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
        )
        dp_fitting = EnerFitting(descrpt=dp_descrpt, neuron=self.n_neuron)
        return EnerModel(
            dp_descrpt,
            dp_fitting,
            type_map=self.type_map,
            data_stat_nbatch=self.data_stat_nbatch,
        )

    def _get_dp_loss(self):
        return EnerStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f,
        )

    def _get_dp_lr(self):
        return LearningRateExp(
            start_lr=self.start_lr, stop_lr=self.stop_lr, decay_steps=self.decay_steps
        )

    def _get_dp_placeholders(self, dataset):
        place_holders = {}
        data_dict = dataset.get_data_dict()
        for kk in data_dict.keys():
            if kk == "type":
                continue
            prec = tf.float64
            place_holders[kk] = tf.placeholder(prec, [None], name="t_" + kk)
            place_holders["find_" + kk] = tf.placeholder(
                tf.float32, name="t_find_" + kk
            )
        place_holders["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        place_holders["natoms_vec"] = tf.placeholder(
            tf.int32, [self.ntypes + 2], name="t_natoms"
        )
        place_holders["default_mesh"] = tf.placeholder(tf.int32, [None], name="t_mesh")
        place_holders["is_training"] = tf.placeholder(tf.bool)
        return place_holders

    def _get_feed_dict(self, batch, place_holders):
        feed_dict = {}
        for kk in batch.keys():
            if kk == "find_type" or kk == "type":
                continue
            if "find_" in kk:
                feed_dict[place_holders[kk]] = batch[kk]
            else:
                feed_dict[place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ["type"]:
            feed_dict[place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ["natoms_vec", "default_mesh"]:
            feed_dict[place_holders[ii]] = batch[ii]
        feed_dict[place_holders["is_training"]] = True
        return feed_dict


class TestEnergy(unittest.TestCase):
    def setUp(self):
        self.dp_trainer = DpTrainer()
        self.wanted_step = 0
        for key in dir(self.dp_trainer):
            if not key.startswith("_") or key == "get_intermediate_state":
                value = getattr(self.dp_trainer, key)
                setattr(self, key, value)

    def test_consistency(self):
        batch, head_dict, stat_dict, vs_dict = self.dp_trainer.get_intermediate_state(
            self.wanted_step
        )
        # Build DeePMD graph
        my_ds = DpLoaderSet(
            self.systems,
            self.batch_size,
            model_params={
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": self.sel,
                    "rcut": self.rcut,
                },
                "type_map": self.type_map,
            },
        )
        sampled = make_stat_input(
            my_ds.systems, my_ds.dataloaders, self.data_stat_nbatch
        )
        my_model = get_model(
            model_params={
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": self.sel,
                    "rcut_smth": self.rcut_smth,
                    "rcut": self.rcut,
                    "neuron": self.filter_neuron,
                    "axis_neuron": self.axis_neuron,
                },
                "fitting_net": {"neuron": self.n_neuron},
                "data_stat_nbatch": self.data_stat_nbatch,
                "type_map": self.type_map,
            },
            sampled=sampled,
        )
        my_model.to(DEVICE)
        my_lr = MyLRExp(self.start_lr, self.stop_lr, self.decay_steps, self.stop_steps)
        my_loss = EnergyStdLoss(
            starter_learning_rate=self.start_lr,
            start_pref_e=self.start_pref_e,
            limit_pref_e=self.limit_pref_e,
            start_pref_f=self.start_pref_f,
            limit_pref_f=self.limit_pref_f,
        )

        # Keep statistics consistency between 2 implentations
        my_em = my_model.descriptor
        mean = stat_dict["descriptor.mean"].reshape([self.ntypes, my_em.get_nsel(), 4])
        stddev = stat_dict["descriptor.stddev"].reshape(
            [self.ntypes, my_em.get_nsel(), 4]
        )
        my_em.set_stat_mean_and_stddev(
            torch.tensor(mean, device=DEVICE),
            torch.tensor(stddev, device=DEVICE),
        )
        my_model.fitting_net.bias_atom_e = torch.tensor(
            stat_dict["fitting_net.bias_atom_e"], device=DEVICE
        )

        # Keep parameter value consistency between 2 implentations
        for name, param in my_model.named_parameters():
            name = name.replace("sea.", "")
            var_name = torch2tf(name)
            var = vs_dict[var_name].value
            with torch.no_grad():
                src = torch.from_numpy(var)
                dst = param.data
                # print(name)
                # print(src.mean(), src.std())
                # print(dst.mean(), dst.std())
                dst.copy_(src)
        # Start forward computing
        batch = my_ds.systems[0]._data_system.preprocess(batch)
        batch["coord"].requires_grad_(True)
        batch["natoms"] = torch.tensor(
            batch["natoms_vec"], device=batch["coord"].device
        ).unsqueeze(0)
        model_predict = my_model(
            batch["coord"], batch["atype"], batch["box"], do_atomic_virial=True
        )
        model_predict_1 = my_model(
            batch["coord"], batch["atype"], batch["box"], do_atomic_virial=False
        )
        p_energy, p_force, p_virial, p_atomic_virial = (
            model_predict["energy"],
            model_predict["force"],
            model_predict["virial"],
            model_predict["atomic_virial"],
        )
        cur_lr = my_lr.value(self.wanted_step)
        model_pred = {
            "energy": p_energy,
            "force": p_force,
        }
        label = {
            "energy": batch["energy"],
            "force": batch["force"],
        }
        loss, _ = my_loss(model_pred, label, int(batch["natoms"][0, 0]), cur_lr)
        np.testing.assert_allclose(
            head_dict["energy"], p_energy.view(-1).cpu().detach().numpy()
        )
        np.testing.assert_allclose(
            head_dict["force"],
            p_force.view(*head_dict["force"].shape).cpu().detach().numpy(),
        )
        rtol = 1e-5
        atol = 1e-8
        np.testing.assert_allclose(
            head_dict["loss"], loss.cpu().detach().numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            head_dict["virial"],
            p_virial.view(*head_dict["virial"].shape).cpu().detach().numpy(),
        )
        np.testing.assert_allclose(
            head_dict["virial"],
            model_predict_1["virial"]
            .view(*head_dict["virial"].shape)
            .cpu()
            .detach()
            .numpy(),
        )
        self.assertIsNone(model_predict_1.get("atomic_virial", None))
        np.testing.assert_allclose(
            head_dict["atomic_virial"],
            p_atomic_virial.view(*head_dict["atomic_virial"].shape)
            .cpu()
            .detach()
            .numpy(),
        )
        optimizer = torch.optim.Adam(my_model.parameters(), lr=cur_lr)
        optimizer.zero_grad()

        def step(step_id):
            bdata = self.training_data.get_trainning_batch()
            optimizer.zero_grad()

        # Compare gradient for consistency
        loss.backward()

        for name, param in my_model.named_parameters():
            name = name.replace("sea.", "")
            var_name = torch2tf(name)
            var_grad = vs_dict[var_name].gradient
            param_grad = param.grad.cpu()
            var_grad = torch.tensor(var_grad)
            assert np.allclose(var_grad, param_grad, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
