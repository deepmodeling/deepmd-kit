# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tasks of an MoE model in training and fine-tuning."""

import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import torch

from deepmd.dpmodel.model.expert_merge import (
    branch_local_parameters,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pt_expt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.model_preset import (
    expand_model_preset,
)

from .compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)

_PT_DATA = str(Path(__file__).parent.parent / "pt" / "water" / "data" / "data_0")
# Every frame uses the weights of its elements, so that two steps train them.
_MOE = {"n_experts": 2, "mat_prob": 0.0}
_DESCRIPTOR = {
    "sel": 40,
    "seed": 1,
    "use_amp": False,
    "add_chg_spin_ebd": True,
    "default_chg_spin": [0.0, 1.0],
    "moe": _MOE,
}
_FITTING = {"seed": 1, "moe": _MOE}
# A fitting network without MoE, which every branch owns.
_PLAIN_FITTING = {"seed": 1}
_LOSS = {"type": "ener", "start_pref_e": 1, "limit_pref_e": 1, "start_pref_f": 1}
_DATA = {"training_data": {"systems": [_PT_DATA], "batch_size": 1}}
_TRAINING = {"numb_steps": 2, "seed": 10, "disp_freq": 1, "save_freq": 2}


def single_task_config(task: str, fitting: dict = _FITTING) -> dict:
    return {
        "model": {
            "preset": "dpa4.5-nano-v20261111",
            "type_map": ["O", "H"],
            "task": task,
            "descriptor": deepcopy(_DESCRIPTOR),
            "fitting_net": deepcopy(fitting),
            "data_stat_nbatch": 1,
        },
        "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5},
        "loss": deepcopy(_LOSS),
        "optimizer": {"type": "HybridMuon", "weight_decay": 0.03},
        "training": {**deepcopy(_DATA), **_TRAINING},
    }


def multi_task_config(branches: dict[str, dict], fitting: dict = _FITTING) -> dict:
    """Branches sharing the descriptor; values extend a branch.

    An MoE fitting network is shared as well, and one without MoE is owned
    by every branch.
    """
    shared_fitting = "moe" in fitting
    return {
        "model": {
            "preset": "dpa4.5-nano-v20261111",
            "shared_dict": {
                "type_map": ["O", "H"],
                "descriptor": deepcopy(_DESCRIPTOR),
                **({"fitting": deepcopy(fitting)} if shared_fitting else {}),
            },
            "model_dict": {
                key: {
                    "type_map": "type_map",
                    "descriptor": "descriptor",
                    "fitting_net": "fitting" if shared_fitting else deepcopy(fitting),
                    "data_stat_nbatch": 1,
                    **extra,
                }
                for key, extra in branches.items()
            },
        },
        "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 1e-5},
        "loss_dict": {key: deepcopy(_LOSS) for key in branches},
        "optimizer": {"type": "HybridMuon", "weight_decay": 0.03},
        "training": {
            "model_prob": dict.fromkeys(branches, 1.0),
            "data_dict": {key: deepcopy(_DATA) for key in branches},
            **_TRAINING,
        },
    }


def build_trainer(
    config: dict,
    finetune_model: str | None = None,
    *,
    restart_model: str | None = None,
    init_model: str | None = None,
):
    config = deepcopy(config)
    config["model"] = expand_model_preset(config["model"])
    multi_task = "model_dict" in config["model"]
    shared_links = None
    if multi_task:
        config["model"], shared_links = preprocess_shared_params(config["model"])
    finetune_links = None
    if finetune_model is not None:
        config["model"], finetune_links = get_finetune_rules(
            finetune_model, config["model"]
        )
    config = update_deepmd_input(config, warning=False)
    config = normalize(config, multi_task=multi_task)
    return get_trainer(
        config,
        init_model=init_model,
        restart_model=restart_model,
        finetune_model=finetune_model,
        finetune_links=finetune_links,
        shared_links=shared_links,
    )


def task_state(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in branch_local_parameters(model)]


class TestExpertTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(_PT_DATA):
            raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")

    def setUp(self) -> None:
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp(prefix="expert_tasks_")
        os.chdir(self.tmpdir)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _check_task_gradients_own_their_storage(self, compiled: bool) -> None:
        """Scaling the gradients of a step in place scales each one once.

        The shared weight and the task increment of an MoE map receive slices
        of one gradient buffer unless the trainer separates them; clipping in
        place would then scale both twice.
        """
        config = multi_task_config({"water": {}, "ice": {}})
        config["training"]["enable_compile"] = compiled
        trainer = build_trainer(config)
        try:
            task = next(iter(trainer.training_tasks))
            input_dict, label_dict = trainer.get_data(is_train=True, task_key=task.key)
            _, loss, _ = trainer.wrapper(
                **input_dict, cur_lr=1e-3, label=label_dict, task_key=task.key
            )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in trainer.wrapper.parameters()
                if parameter.grad is not None
            ]
            expected = [0.5 * gradient for gradient in gradients]
            torch._foreach_mul_(gradients, 0.5)
            for gradient, reference in zip(gradients, expected, strict=True):
                torch.testing.assert_close(gradient, reference, rtol=0.0, atol=0.0)
        finally:
            trainer._close_data_systems()

    def test_task_gradients_own_their_storage(self) -> None:
        self._check_task_gradients_own_their_storage(compiled=False)

    @REQUIRES_SUPPORTED_COMPILE
    def test_compiled_task_gradients_own_their_storage(self) -> None:
        self._check_task_gradients_own_their_storage(compiled=True)

    def test_tasks_extend_in_finetuning(self) -> None:
        # === A model trained alone keeps the weights of its task fixed ===
        trainer = build_trainer(single_task_config("water"))
        model = trainer.models["Default"]
        task = branch_local_parameters(model)
        self.assertTrue(task)
        self.assertFalse(any(parameter.requires_grad for parameter in task))
        task_ids = {id(parameter) for parameter in task}
        # The parameters that stay fixed next to the task.
        fixed = {
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and id(parameter) not in task_ids
        }
        trainer.run()
        self.assertFalse(any(bool(parameter.any()) for parameter in task))
        trained = {
            name
            for name, parameter in model.named_parameters()
            if bool(parameter.any())
        }
        for key in ("element_weight", "expert_weight"):
            self.assertTrue(any(key in name for name in trained), key)

        # === Give the task weights of its own, as a branch of several has ===
        checkpoint = os.path.join(self.tmpdir, "model.ckpt.pt")
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        generator = torch.Generator(device="cpu").manual_seed(3)
        task_names = [
            name
            for name in state["model"]
            if "task_weight" in name or "task_bias" in name or "adam_task" in name
        ]
        self.assertEqual(len(task_names), len(task))
        for name in task_names:
            value = state["model"][name]
            state["model"][name] = 0.05 * torch.randn(
                value.shape, generator=generator
            ).to(value)
        source = {name: state["model"][name] for name in task_names}
        torch.save(state, checkpoint)

        # === Three tasks from the one: continued, started from it, and new ===
        trainer = build_trainer(
            multi_task_config(
                {"water": {}, "ice": {"finetune_head": "water"}, "vapor": {}}
            ),
            finetune_model=checkpoint,
        )
        state = trainer._unwrapped.state_dict()
        for name, value in source.items():
            for branch in ("water", "ice"):
                torch.testing.assert_close(
                    state[name.replace("model.Default.", f"model.{branch}.")], value
                )
            # A new task starts from the shared weights alone.
            fresh = state[name.replace("model.Default.", "model.vapor.")]
            self.assertFalse(bool(fresh.any()), name)
        # The shared networks of the descriptor and of the fitting are those
        # of the pretrained model in every branch.
        shared = [
            name
            for name in state
            if name.startswith(("model.vapor.", "model.water."))
            and "element_weight" in name
        ]
        for part in (".descriptor.", ".fitting_net."):
            self.assertTrue(any(part in name for name in shared), part)
        pretrained = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        for name in shared:
            branch = name.split(".")[1]
            torch.testing.assert_close(
                state[name],
                pretrained["model"][name.replace(f"model.{branch}.", "model.Default.")],
            )
        # Several tasks train the weights of their own, and a single task
        # fixes nothing else.
        before = {key: task_state(trainer.models[key]) for key in trainer.model_keys}
        for key in trainer.model_keys:
            self.assertTrue(
                all(
                    p.requires_grad
                    for p in branch_local_parameters(trainer.models[key])
                )
            )
            self.assertEqual(
                {
                    name
                    for name, parameter in trainer.models[key].named_parameters()
                    if not parameter.requires_grad
                },
                fixed,
            )
        trainer.run()
        moved = {
            key: any(
                not torch.equal(old, new)
                for old, new in zip(
                    before[key], task_state(trainer.models[key]), strict=True
                )
            )
            for key in trainer.model_keys
        }
        self.assertTrue(any(moved.values()), moved)

    def test_new_task_initializes_a_fitting_without_moe(self) -> None:
        trainer = build_trainer(single_task_config("water", _PLAIN_FITTING))
        trainer.run()
        checkpoint = os.path.join(self.tmpdir, "model.ckpt.pt")
        pretrained = torch.load(checkpoint, map_location=DEVICE, weights_only=True)[
            "model"
        ]
        trainer = build_trainer(
            multi_task_config({"water": {}, "vapor": {}}, _PLAIN_FITTING),
            finetune_model=checkpoint,
        )
        state = trainer._unwrapped.state_dict()
        networks = [
            name
            for name in pretrained
            if ".fitting_net.nets." in name and name.endswith(".w")
        ]
        self.assertTrue(networks)
        for name in networks:
            # The continued task keeps its fitting network, the new one does not.
            torch.testing.assert_close(
                state[name.replace("model.Default.", "model.water.")], pretrained[name]
            )
            self.assertFalse(
                torch.equal(
                    state[name.replace("model.Default.", "model.vapor.")],
                    pretrained[name],
                )
            )
        # Both tasks take the descriptor, and the new one starts without a
        # task of its own.
        task = branch_local_parameters(trainer.models["vapor"])
        self.assertTrue(task)
        self.assertFalse(any(bool(parameter.any()) for parameter in task))
        for name, value in pretrained.items():
            if ".descriptor." in name and "element_weight" in name:
                torch.testing.assert_close(
                    state[name.replace("model.Default.", "model.vapor.")], value
                )

    def test_an_unmerged_model_continues_training(self) -> None:
        """A checkpoint that still holds MoE restarts and initializes a model.

        MoE is part of the model, so a checkpoint of a model that has it holds
        the increments and the router along with the shared weights, and a
        training that starts from it continues the MoE model rather than a
        plain one.
        """
        trainer = build_trainer(single_task_config("water"))
        trainer.run()
        checkpoint = os.path.join(self.tmpdir, "model.ckpt.pt")
        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)["model"]
        expert = {
            name
            for name in saved
            if any(
                key in name
                for key in ("task_weight", "element_weight", "expert_weight")
            )
        }
        self.assertTrue(expert)
        self.assertTrue(any(bool(saved[name].any()) for name in expert))
        self.assertTrue(any("expert_router" in name for name in saved))

        for mode in ("restart_model", "init_model"):
            resumed = build_trainer(single_task_config("water"), **{mode: checkpoint})
            state = resumed._unwrapped.state_dict()
            for name in expert:
                torch.testing.assert_close(state[name], saved[name], msg=name)
            resumed.run()

    def test_a_multi_task_model_continues_training(self) -> None:
        """A multi-task checkpoint keeps the MoE of every branch.

        The branches share the descriptor, so they share its MoE as well; a
        restart and an initialization restore those weights along with the
        task increments of every branch.
        """
        branches = {"water": {}, "vapor": {}}
        trainer = build_trainer(multi_task_config(branches, _PLAIN_FITTING))
        trainer.run()
        checkpoint = os.path.join(self.tmpdir, "model.ckpt.pt")
        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)["model"]
        expert = {
            name
            for name in saved
            if any(
                key in name
                for key in ("task_weight", "element_weight", "expert_weight")
            )
        }
        self.assertTrue(expert)
        for mode in ("restart_model", "init_model"):
            resumed = build_trainer(
                multi_task_config(branches, _PLAIN_FITTING), **{mode: checkpoint}
            )
            self.assertEqual(set(resumed.model_keys), set(branches))
            state = resumed._unwrapped.state_dict()
            for name in expert:
                torch.testing.assert_close(state[name], saved[name], msg=name)
            resumed.run()
