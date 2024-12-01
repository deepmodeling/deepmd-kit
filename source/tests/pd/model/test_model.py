# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from typing import (
    NamedTuple,
)

import numpy as np
import paddle
import tensorflow.compat.v1 as tf

from deepmd.pd.utils import (
    env,
)

tf.disable_eager_execution()

from pathlib import (
    Path,
)

from deepmd.dpmodel.utils.learning_rate import LearningRateExp as MyLRExp
from deepmd.pd.loss import (
    EnergyStdLoss,
)
from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pd.utils.env import (
    DEVICE,
)
from deepmd.tf.common import (
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

from ..test_finetune import (
    energy_data_requirement,
)


class VariableState(NamedTuple):
    value: np.ndarray
    gradient: np.ndarray


def paddle2tf(paddle_name, last_layer_id=None):
    fields = paddle_name.split(".")
    offset = int(fields[3] == "networks") + 1
    element_id = int(fields[2 + offset])
    if fields[1] == "descriptor":
        if fields[2].startswith("compress_"):
            return None
        layer_id = int(fields[4 + offset]) + 1
        weight_type = fields[5 + offset]
        ret = f"filter_type_all/{weight_type}_{layer_id}_{element_id}:0"
    elif fields[1] == "fitting_net":
        layer_id = int(fields[4 + offset])
        weight_type = fields[5 + offset]
        if layer_id != last_layer_id:
            ret = f"layer_{layer_id}_type_{element_id}/{weight_type}:0"
        else:
            ret = f"final_layer_type_{element_id}/{weight_type}:0"
    else:
        raise RuntimeError(f"Unexpected parameter name: {paddle_name}")
    return ret


class DpTrainer:
    def __init__(self) -> None:
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
        dp_ds.add_data_requirements(dp_model.input_requirement)
        dp_ds.add_data_requirements(dp_loss.label_requirement)
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
                "atom_virial": model_pred["atom_virial"],
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

        tf.reset_default_graph()
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
        return data

    def _get_dp_model(self):
        dp_descrpt = DescrptSeA_tf(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            neuron=self.filter_neuron,
            axis_neuron=self.axis_neuron,
        )
        dp_fitting = EnerFitting(
            dp_descrpt.get_ntypes(), dp_descrpt.get_dim_out(), neuron=self.n_neuron
        )
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
    def setUp(self) -> None:
        self.dp_trainer = DpTrainer()
        self.wanted_step = 0
        for key in dir(self.dp_trainer):
            if not key.startswith("_") or key == "get_intermediate_state":
                value = getattr(self.dp_trainer, key)
                setattr(self, key, value)

    def test_consistency(self) -> None:
        batch, head_dict, stat_dict, vs_dict = self.dp_trainer.get_intermediate_state(
            self.wanted_step
        )
        # Build DeePMD graph
        my_ds = DpLoaderSet(self.systems, self.batch_size, self.type_map)
        my_ds.add_data_requirement(energy_data_requirement)
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
                "fitting_net": {"neuron": self.n_neuron, "mixed_types": False},
                "data_stat_nbatch": self.data_stat_nbatch,
                "type_map": self.type_map,
            },
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

        # Keep statistics consistency between 2 implementations
        my_em = my_model.get_descriptor()
        mean = stat_dict["descriptor.mean"].reshape([self.ntypes, my_em.get_nsel(), 4])
        stddev = stat_dict["descriptor.stddev"].reshape(
            [self.ntypes, my_em.get_nsel(), 4]
        )
        my_em.set_stat_mean_and_stddev(
            paddle.to_tensor(mean).to(device=DEVICE),
            paddle.to_tensor(stddev).to(device=DEVICE),
        )
        my_model.get_fitting_net().bias_atom_e = paddle.to_tensor(
            stat_dict["fitting_net.bias_atom_e"], place=DEVICE
        )

        # Keep parameter value consistency between 2 implementations
        for name, param in my_model.named_parameters():
            name = name.replace("sea.", "")
            var_name = paddle2tf(name, last_layer_id=len(self.n_neuron))
            if var_name is None:
                continue
            var = vs_dict[var_name].value
            with paddle.no_grad():
                src = paddle.to_tensor(var)
                dst = param
                # print(name)
                # print(src.mean(), src.std())
                # print(dst.mean(), dst.std())
                paddle.assign(src, dst)
        # Start forward computing
        tmp = np.copy(batch["natoms_vec"])
        batch = my_ds.systems[0]._data_system._get_subdata(batch, 0)
        batch = my_ds.systems[0]._data_system.reformat_data_torch(batch)
        for key in ["coord", "atype", "box", "energy", "force"]:
            batch[key] = paddle.to_tensor(batch[key]).to(device=env.DEVICE)
            batch[key] = batch[key].unsqueeze(0)
        batch["coord"].stop_gradient = False
        batch["natoms_vec"] = tmp
        batch["natoms"] = paddle.to_tensor(
            batch["natoms_vec"], place=batch["coord"].place
        ).unsqueeze(0)
        model_input = {
            "coord": batch["coord"].to(env.DEVICE),
            "atype": batch["atype"].to(env.DEVICE),
            "box": batch["box"].to(env.DEVICE),
            "do_atomic_virial": True,
        }
        model_input_1 = {
            "coord": batch["coord"].to(env.DEVICE),
            "atype": batch["atype"].to(env.DEVICE),
            "box": batch["box"].to(env.DEVICE),
            "do_atomic_virial": False,
        }
        label = {
            "energy": batch["energy"].to(env.DEVICE),
            "find_energy": 1.0,
            "force": batch["force"].to(env.DEVICE),
            "find_force": 1.0,
        }
        cur_lr = my_lr.value(self.wanted_step)
        model_predict, loss, _ = my_loss(
            model_input, my_model, label, int(batch["natoms"][0, 0]), cur_lr
        )
        model_predict_1 = my_model(**model_input_1)
        p_energy, p_force, p_virial, p_atomic_virial = (
            model_predict["energy"],
            model_predict["force"],
            model_predict["virial"],
            model_predict["atom_virial"],
        )
        np.testing.assert_allclose(
            head_dict["energy"], p_energy.reshape([-1]).cpu().detach().numpy()
        )
        np.testing.assert_allclose(
            head_dict["force"],
            p_force.reshape(head_dict["force"].shape).cpu().detach().numpy(),
        )
        rtol = 1e-5
        atol = 1e-8
        np.testing.assert_allclose(
            head_dict["loss"], loss.cpu().detach().numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            head_dict["virial"],
            p_virial.reshape(head_dict["virial"].shape).cpu().detach().numpy(),
        )
        np.testing.assert_allclose(
            head_dict["virial"],
            model_predict_1["virial"]
            .reshape([*head_dict["virial"].shape])
            .cpu()
            .detach()
            .numpy(),
        )
        self.assertIsNone(model_predict_1.get("atom_virial", None))
        np.testing.assert_allclose(
            head_dict["atom_virial"],
            p_atomic_virial.reshape(head_dict["atom_virial"].shape)
            .cpu()
            .detach()
            .numpy(),
        )
        optimizer = paddle.optimizer.Adam(cur_lr, parameters=my_model.parameters())
        optimizer.clear_grad()

        def step(step_id) -> None:
            bdata = self.training_data.get_trainning_batch()
            optimizer.clear_grad()

        # Compare gradient for consistency
        loss.backward()

        for name, param in my_model.named_parameters():
            name = name.replace("sea.", "")
            var_name = paddle2tf(name, last_layer_id=len(self.n_neuron))
            if var_name is None:
                continue
            var_grad = vs_dict[var_name].gradient
            param_grad = param.grad.cpu()
            var_grad = paddle.to_tensor(var_grad).to(device="cpu")
            assert np.allclose(var_grad, param_grad, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
