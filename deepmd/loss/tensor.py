import numpy as np

from deepmd.common import add_data_requirement
from deepmd.env import global_cvt_2_tf_float
from deepmd.env import paddle
from deepmd.env import tf
from deepmd.utils.sess import run_sess

from .loss import Loss


class TensorLoss(Loss):
    """Loss function for tensorial properties."""

    def __init__(self, jdata, **kwarg):
        model = kwarg.get("model", None)
        if model is not None:
            self.type_sel = model.get_sel_type()
        else:
            self.type_sel = None
        self.tensor_name = kwarg["tensor_name"]
        self.tensor_size = kwarg["tensor_size"]
        self.label_name = kwarg["label_name"]
        if jdata is not None:
            self.scale = jdata.get("scale", 1.0)
        else:
            self.scale = 1.0

        # YHT: added for global / local dipole combination
        assert jdata is not None, "Please provide loss parameters!"
        # YWolfeee: modify, use pref / pref_atomic, instead of pref_weight / pref_atomic_weight
        self.local_weight = jdata.get("pref_atomic", None)
        self.global_weight = jdata.get("pref", None)

        assert (
            self.local_weight is not None and self.global_weight is not None
        ), "Both `pref` and `pref_atomic` should be provided."
        assert (
            self.local_weight >= 0.0 and self.global_weight >= 0.0
        ), "Can not assign negative weight to `pref` and `pref_atomic`"
        assert (self.local_weight > 0.0) or (self.global_weight > 0.0), AssertionError(
            "Can not assian zero weight both to `pref` and `pref_atomic`"
        )

        # data required
        add_data_requirement(
            "atomic_" + self.label_name,
            self.tensor_size,
            atomic=True,
            must=False,
            high_prec=False,
            type_sel=self.type_sel,
        )
        add_data_requirement(
            self.label_name,
            self.tensor_size,
            atomic=False,
            must=False,
            high_prec=False,
            type_sel=self.type_sel,
        )

    def compute_loss(self, learning_rate, natoms, model_dict, label_dict, suffix):
        polar_hat = label_dict[self.label_name]
        atomic_polar_hat = label_dict["atomic_" + self.label_name]
        polar = paddle.reshape(model_dict[self.tensor_name], [-1])

        find_global = label_dict["find_" + self.label_name]
        find_atomic = label_dict["find_atomic_" + self.label_name]

        # YHT: added for global / local dipole combination
        l2_loss = 0.0
        more_loss = {
            "local_loss": 0.0,
            "global_loss": 0.0,
        }

        if self.local_weight > 0.0:
            local_loss = find_atomic * paddle.mean(
                paddle.square(self.scale * (polar - atomic_polar_hat)),
                name="l2_" + suffix,
            )
            more_loss["local_loss"] = local_loss
            l2_loss += self.local_weight * local_loss
            # self.l2_loss_local_summary = paddle.summary.scalar(
            #     "l2_local_loss_" + suffix, paddle.sqrt(more_loss["local_loss"])
            # )

        if self.global_weight > 0.0:  # Need global loss
            atoms = 0
            if self.type_sel is not None:
                for w in self.type_sel:
                    atoms += natoms[2 + w]
            else:
                atoms = natoms[0]
            nframes = paddle.shape(polar)[0] // self.tensor_size // atoms
            # get global results
            global_polar = paddle.reshape(
                paddle.sum(
                    paddle.reshape(polar, [nframes, -1, self.tensor_size]), axis=1
                ),
                [-1],
            )
            # if self.atomic: # If label is local, however
            #    global_polar_hat = paddle.reshape(paddle.sum(paddle.reshape(
            #        polar_hat, [nframes, -1, self.tensor_size]), axis=1),[-1])
            # else:
            #    global_polar_hat = polar_hat

            global_loss = find_global * paddle.mean(
                paddle.square(self.scale * (global_polar - polar_hat)),
                name="l2_" + suffix,
            )

            more_loss["global_loss"] = global_loss
            # self.l2_loss_global_summary = paddle.summary.scalar(
            #     "l2_global_loss_" + suffix,
            #     paddle.sqrt(more_loss["global_loss"]) / global_cvt_2_tf_float(atoms),
            # )

            # YWolfeee: should only consider atoms with dipole, i.e. atoms
            # atom_norm  = 1./ global_cvt_2_tf_float(natoms[0])
            atom_norm = 1.0 / atoms
            global_loss *= atom_norm

            l2_loss += self.global_weight * global_loss

        self.l2_more = more_loss
        self.l2_l = l2_loss

        # self.l2_loss_summary = paddle.summary.scalar("l2_loss_" + suffix, tf.sqrt(l2_loss))
        return l2_loss, more_loss

    def eval(self, model, batch_data, natoms):
        atoms = 0
        if self.type_sel is not None:
            for w in self.type_sel:
                atoms += natoms[2 + w]
        else:
            atoms = natoms[0]

        model_inputs = {}
        for kk in batch_data.keys():
            if kk == "find_type" or kk == "type":
                continue
            prec = "float64"
            if "find_" in kk:
                model_inputs[kk] = paddle.to_tensor(batch_data[kk], dtype="float64")
            else:
                model_inputs[kk] = paddle.to_tensor(
                    np.reshape(batch_data[kk], [-1]), dtype=prec
                )

        for ii in ["type"]:
            model_inputs[ii] = paddle.to_tensor(
                np.reshape(batch_data[ii], [-1]), dtype="int32"
            )
        for ii in ["natoms_vec", "default_mesh"]:
            model_inputs[ii] = paddle.to_tensor(batch_data[ii], dtype="int32")
        model_inputs["is_training"] = paddle.to_tensor(False)
        model_inputs["natoms_vec"] = paddle.to_tensor(
            model_inputs["natoms_vec"], place="cpu"
        )

        model_pred = model(
            model_inputs["coord"],
            model_inputs["type"],
            model_inputs["natoms_vec"],
            model_inputs["box"],
            model_inputs["default_mesh"],
            model_inputs,
            suffix="",
            reuse=False,
        )

        l2_l, l2_more = self.compute_loss(
            # 0.0, natoms, model_dict, batch_data
            0.0,
            model_inputs["natoms_vec"],
            model_pred,
            model_inputs,
            suffix="test",
        )

        run_data = [
            (float(l2_l)),
            (float(l2_more["local_loss"])),
            (float(l2_more["global_loss"])),
        ]
        error, error_lc, error_gl = run_data

        results = {"natoms": atoms, "rmse": np.sqrt(error)}
        if self.local_weight > 0.0:
            results["rmse_lc"] = np.sqrt(error_lc)
        if self.global_weight > 0.0:
            results["rmse_gl"] = np.sqrt(error_gl) / atoms
        return results
