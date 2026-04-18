# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for distributed (DDP) training in the pt_expt backend.

Uses ``torch.multiprocessing.spawn`` with auto-detected backend
(``nccl`` on CUDA, ``gloo`` on CPU).

Verifies that:
1. Single-task DDP training completes and produces correct outputs
2. Multi-task DDP training completes and produces correct outputs
3. DDP gradient averaging matches manual average of per-rank gradients
4. Multi-task DDP gradient averaging works correctly
5. Finetune + DDP: selective weight copy via _unwrapped
6. Finetune + DDP with random fitting: descriptor from pretrained, fitting random
7. Finetune + DDP with new type: exercises _unwrapped.model["Default"] + stat broadcast
8. DDP + torch.compile: single-task and multi-task compile under DDP
"""

import os
import shutil
import socket
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
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

# Paths to the water data used by PT tests
_PT_DATA = str(Path(__file__).parent.parent / "pt" / "water" / "data" / "data_0")

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "water",
)

# Auto-detect DDP backend based on device availability.
_DDP_BACKEND = "nccl" if torch.cuda.is_available() else "gloo"


def _find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_config(data_dir: str, numb_steps: int = 2) -> dict:
    """Build a minimal single-task config."""
    return {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [6, 12],
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [8, 16],
                "resnet_dt": False,
                "axis_neuron": 4,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [16, 16],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [data_dir],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [data_dir],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }


def _make_multitask_config(data_dir: str, numb_steps: int = 2) -> dict:
    """Build a minimal multi-task config with shared descriptor."""
    descriptor = {
        "type": "se_e2_a",
        "sel": [6, 12],
        "rcut_smth": 0.50,
        "rcut": 3.00,
        "neuron": [8, 16],
        "resnet_dt": False,
        "axis_neuron": 4,
        "type_one_side": True,
        "seed": 1,
    }
    fitting = {
        "neuron": [16, 16],
        "resnet_dt": True,
        "seed": 1,
    }
    return {
        "model": {
            "shared_dict": {
                "my_type_map": ["O", "H"],
                "my_descriptor": deepcopy(descriptor),
            },
            "model_dict": {
                "model_1": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": deepcopy(fitting),
                    "data_stat_nbatch": 1,
                },
                "model_2": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": deepcopy(fitting),
                    "data_stat_nbatch": 1,
                },
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss_dict": {
            "model_1": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "model_2": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
        },
        "training": {
            "model_prob": {
                "model_1": 0.5,
                "model_2": 0.5,
            },
            "data_dict": {
                "model_1": {
                    "stat_file": "./stat_files/model_1",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
                "model_2": {
                    "stat_file": "./stat_files/model_2",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }


def _make_dpa1_config(data_dir: str, numb_steps: int = 2) -> dict:
    """Build a minimal DPA1 config (mixed_types) for finetune new-type tests."""
    return {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa1",
                "sel": 12,
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [4, 8],
                "axis_neuron": 4,
                "attn": 4,
                "attn_layer": 1,
                "attn_dotr": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [8, 8],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": [data_dir],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [data_dir],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }


def _subsample_data(src_dir: str, dst_dir: str, nframes: int = 2) -> None:
    """Copy a data system, keeping only the first *nframes* frames."""
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    set_dir = os.path.join(dst_dir, "set.000")
    for name in os.listdir(set_dir):
        if name.endswith(".npy"):
            arr = np.load(os.path.join(set_dir, name))
            np.save(os.path.join(set_dir, name), arr[:nframes])


# ---------------------------------------------------------------------------
# Worker functions for mp.spawn
# ---------------------------------------------------------------------------


def _worker_single_task_train(rank, world_size, port, data_dir, result_dict):
    """Worker: run single-task DDP training."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_st_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_config(data_dir, numb_steps=2)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)
            trainer.run()

            # Collect results
            lcurve_exists = os.path.exists("lcurve.out")
            ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]

            # Get final weights
            weights = {
                name: p.detach().cpu().clone()
                for name, p in trainer._unwrapped.named_parameters()
            }

            result_dict[rank] = {
                "lcurve_exists": lcurve_exists,
                "num_ckpts": len(ckpt_files),
                "weights": weights,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_multitask_train(rank, world_size, port, data_dir, result_dict):
    """Worker: run multi-task DDP training."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_mt_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_multitask_config(data_dir, numb_steps=2)
            config["model"], shared_links = preprocess_shared_params(config["model"])
            config = update_deepmd_input(config, warning=False)
            config = normalize(config, multi_task=True)
            trainer = get_trainer(config, shared_links=shared_links)
            trainer.run()

            lcurve_exists = os.path.exists("lcurve.out")
            ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]

            # Get shared descriptor params from model_1
            desc_params = {}
            for name, p in trainer._unwrapped.model[
                "model_1"
            ].atomic_model.descriptor.named_parameters():
                desc_params[name] = p.detach().cpu().clone()

            result_dict[rank] = {
                "lcurve_exists": lcurve_exists,
                "num_ckpts": len(ckpt_files),
                "desc_params": desc_params,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_gradient_test(rank, world_size, port, data_dir, result_dict):
    """Worker: run 1 step of DDP training, collect gradients and input data."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_grad_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_config(data_dir, numb_steps=1)
            config["model"]["descriptor"]["precision"] = "float64"
            config["model"]["fitting_net"]["precision"] = "float64"
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)

            # Run one forward/backward step manually
            trainer.wrapper.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = trainer.get_data(is_train=True, task_key="Default")

            cur_lr_sched = trainer.scheduler.get_last_lr()[0]
            _, loss, _ = trainer.wrapper(
                **input_dict,
                cur_lr=cur_lr_sched,
                label=label_dict,
            )
            loss.backward()  # DDP all-reduces gradients here

            # Collect post-all-reduce gradients
            grads = {}
            for name, p in trainer._unwrapped.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.detach().cpu().clone()

            # Collect input batch (for single-process replay)
            batch = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.detach().cpu().clone()
                else:
                    batch[k] = v
            for k, v in label_dict.items():
                if isinstance(v, torch.Tensor):
                    batch[f"label_{k}"] = v.detach().cpu().clone()

            # Initial model state dict (before any optimizer step)
            init_state = {
                k: v.detach().cpu().clone()
                for k, v in trainer._unwrapped.state_dict().items()
                if k != "_extra_state"
            }

            result_dict[rank] = {
                "grads": grads,
                "batch": batch,
                "init_state": init_state,
                "config": config,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_multitask_gradient_test(rank, world_size, port, data_dir, result_dict):
    """Worker: run 1 step of multi-task DDP training, collect gradients."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_mt_grad_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_multitask_config(data_dir, numb_steps=1)
            config["model"], shared_links = preprocess_shared_params(config["model"])
            config = update_deepmd_input(config, warning=False)
            config = normalize(config, multi_task=True)
            trainer = get_trainer(config, shared_links=shared_links)

            # Run one step with deterministic task selection

            # Force task_key = "model_1" for all ranks (deterministic)
            trainer.wrapper.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = trainer.get_data(is_train=True, task_key="model_1")
            cur_lr_sched = trainer.scheduler.get_last_lr()[0]
            _, loss, _ = trainer.wrapper(
                **input_dict,
                cur_lr=cur_lr_sched,
                label=label_dict,
                task_key="model_1",
            )
            loss.backward()

            grads = {}
            for name, p in trainer._unwrapped.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.detach().cpu().clone()

            result_dict[rank] = {
                "grads": grads,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_check_resume(
    rank, world_size, port, data_dir, ckpt_path, numb_steps, is_restart, result_dict
):
    """Worker: build DDP trainer from checkpoint, capture initial state, then train.

    Parameters
    ----------
    is_restart : bool
        True  → restart_model (continue training, restore optimizer & step).
        False → init_model   (inherit weights, reset step to 0).
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_resume_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_config(data_dir, numb_steps=numb_steps)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)

            if is_restart:
                trainer = get_trainer(config, restart_model=ckpt_path)
            else:
                trainer = get_trainer(config, init_model=ckpt_path)

            # Capture initial state BEFORE training
            init_weights = {
                name: p.detach().cpu().clone()
                for name, p in trainer._unwrapped.named_parameters()
            }
            start_step = trainer.start_step
            init_lr = trainer.scheduler.get_last_lr()[0]

            trainer.run()

            result_dict[rank] = {
                "init_weights": init_weights,
                "start_step": start_step,
                "init_lr": init_lr,
                "lcurve_exists": os.path.exists("lcurve.out"),
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_finetune(
    rank, world_size, port, ckpt_path, config_dict, model_branch, result_dict
):
    """Worker: DDP finetune from checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Absolute path to pretrained checkpoint (.pt).
    config_dict : dict
        Already normalized config with absolute data paths.
    model_branch : str or None
        ``"RANDOM"`` for random fitting, ``None`` for normal.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_ft_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = deepcopy(config_dict)
            config["model"], finetune_links = get_finetune_rules(
                ckpt_path,
                config["model"],
                model_branch=model_branch or "",
            )

            trainer = get_trainer(
                config,
                finetune_model=ckpt_path,
                finetune_links=finetune_links,
            )

            # Capture state after finetune setup (before training)
            init_state = {
                k: v.detach().cpu().clone()
                for k, v in trainer._unwrapped.state_dict().items()
                if k != "_extra_state"
            }

            trainer.run()

            result_dict[rank] = {
                "init_state": init_state,
                "lcurve_exists": os.path.exists("lcurve.out"),
                "ckpt_files": [f for f in os.listdir(".") if f.endswith(".pt")],
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDDPSingleTaskTrain(unittest.TestCase):
    """Smoke test: single-task DDP training with 2 ranks."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = os.path.join(data_dir, "data_0")

    def test_ddp_single_task_trains(self) -> None:
        """2 ranks, se_e2_a, 2 training steps — verify completion and outputs."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_single_task_train,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)

        # Only rank 0 produces lcurve.out and checkpoints
        self.assertTrue(results[0]["lcurve_exists"], "rank 0 should produce lcurve.out")
        self.assertFalse(
            results[1]["lcurve_exists"], "rank 1 should NOT produce lcurve.out"
        )
        self.assertGreater(results[0]["num_ckpts"], 0, "rank 0 should save checkpoints")
        self.assertEqual(
            results[1]["num_ckpts"], 0, "rank 1 should NOT save checkpoints"
        )

        # Final weights should be identical across ranks
        for name in results[0]["weights"]:
            torch.testing.assert_close(
                results[0]["weights"][name],
                results[1]["weights"][name],
                msg=f"Weights differ across ranks: {name}",
            )


class TestDDPMultiTaskTrain(unittest.TestCase):
    """Smoke test: multi-task DDP training with 2 ranks."""

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(_PT_DATA):
            raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")
        cls.data_dir = _PT_DATA

    def test_ddp_multitask_trains(self) -> None:
        """2 ranks, multi-task, 2 steps — verify completion."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_multitask_train,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)

        # Only rank 0 produces output files
        self.assertTrue(results[0]["lcurve_exists"])
        self.assertFalse(results[1]["lcurve_exists"])
        self.assertGreater(results[0]["num_ckpts"], 0)
        self.assertEqual(results[1]["num_ckpts"], 0)

        # Shared descriptor params should be identical across ranks
        for name in results[0]["desc_params"]:
            torch.testing.assert_close(
                results[0]["desc_params"][name],
                results[1]["desc_params"][name],
                msg=f"Shared descriptor param differs across ranks: {name}",
            )


class TestDDPGradientAveraging(unittest.TestCase):
    """Core DDP correctness: gradient averaging matches manual computation.

    Each DDP rank processes different data. After all-reduce, all ranks have
    the averaged gradient. We verify:
    1. Both ranks have identical gradients (DDP guarantee)
    2. The DDP gradient equals the average of per-rank single-process gradients
    """

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = os.path.join(data_dir, "data_0")

    def test_ddp_gradient_equals_average(self) -> None:
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_gradient_test,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # 1. Verify gradients are identical on both ranks (DDP guarantee)
        for name in r0["grads"]:
            self.assertIn(name, r1["grads"], f"Grad key missing on rank 1: {name}")
            torch.testing.assert_close(
                r0["grads"][name],
                r1["grads"][name],
                atol=0,
                rtol=0,
                msg=f"Gradients should be identical across ranks: {name}",
            )

        # 2. Rebuild model in single process, replay each rank's batch,
        #    compute manual average, compare to DDP gradient
        config = r0["config"]
        tmpdir = tempfile.mkdtemp(prefix="ddp_grad_verify_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            trainer = get_trainer(config)
            # Load same initial state as DDP workers
            state_to_load = dict(trainer._unwrapped.state_dict())
            for k in r0["init_state"]:
                state_to_load[k] = r0["init_state"][k]
            trainer._unwrapped.load_state_dict(state_to_load)
            trainer.wrapper.train()

            # Forward+backward with rank 0's batch
            trainer.optimizer.zero_grad(set_to_none=True)
            input_0 = {
                k: v.clone()
                for k, v in r0["batch"].items()
                if not k.startswith("label_")
            }
            label_0 = {
                k[len("label_") :]: v.clone()
                for k, v in r0["batch"].items()
                if k.startswith("label_")
            }
            input_0["coord"] = input_0["coord"].requires_grad_(True)
            cur_lr = trainer.scheduler.get_last_lr()[0]
            _, loss_0, _ = trainer.wrapper(**input_0, cur_lr=cur_lr, label=label_0)
            loss_0.backward()
            grad_0 = {
                name: p.grad.detach().clone()
                for name, p in trainer._unwrapped.named_parameters()
                if p.grad is not None
            }

            # Forward+backward with rank 1's batch
            trainer.optimizer.zero_grad(set_to_none=True)
            input_1 = {
                k: v.clone()
                for k, v in r1["batch"].items()
                if not k.startswith("label_")
            }
            label_1 = {
                k[len("label_") :]: v.clone()
                for k, v in r1["batch"].items()
                if k.startswith("label_")
            }
            input_1["coord"] = input_1["coord"].requires_grad_(True)
            _, loss_1, _ = trainer.wrapper(**input_1, cur_lr=cur_lr, label=label_1)
            loss_1.backward()
            grad_1 = {
                name: p.grad.detach().clone()
                for name, p in trainer._unwrapped.named_parameters()
                if p.grad is not None
            }

            # Expected = average of the two
            for name in r0["grads"]:
                if name in grad_0 and name in grad_1:
                    expected = (grad_0[name] + grad_1[name]) / 2.0
                    torch.testing.assert_close(
                        r0["grads"][name],
                        expected,
                        atol=1e-10,
                        rtol=1e-10,
                        msg=f"DDP grad != avg(rank0, rank1) for {name}",
                    )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestDDPMultiTaskGradient(unittest.TestCase):
    """Verify DDP gradient averaging with multi-task training."""

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(_PT_DATA):
            raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")
        cls.data_dir = _PT_DATA

    def test_ddp_multitask_gradient(self) -> None:
        """Both ranks pick same task; gradients should be identical after all-reduce."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_multitask_gradient_test,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # Gradients should be identical across ranks
        for name in r0["grads"]:
            self.assertIn(name, r1["grads"], f"Grad key missing on rank 1: {name}")
            torch.testing.assert_close(
                r0["grads"][name],
                r1["grads"][name],
                atol=0,
                rtol=0,
                msg=f"Multi-task DDP gradients differ across ranks: {name}",
            )


class _DDPResumeBase(unittest.TestCase):
    """Shared setup: train 2 steps in single process, save checkpoint + weights."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = os.path.join(data_dir, "data_0")

        cls._tmpdir = tempfile.mkdtemp(prefix="ddp_resume_setup_")
        old_cwd = os.getcwd()
        os.chdir(cls._tmpdir)
        try:
            config = _make_config(cls.data_dir, numb_steps=2)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)
            trainer.run()

            cls.ckpt_path = os.path.join(cls._tmpdir, "model.ckpt.pt")
            assert os.path.exists(cls.ckpt_path), "Checkpoint not created"

            # Record phase-1 final weights for comparison
            cls.phase1_weights = {
                name: p.detach().cpu().clone()
                for name, p in trainer.wrapper.named_parameters()
            }
            cls.lr_config = config["learning_rate"].copy()
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmpdir, ignore_errors=True)


class TestDDPInitModel(_DDPResumeBase):
    """DDP init_model: inherits weights but resets step to 0."""

    def test_ddp_init_model(self) -> None:
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_check_resume,
            args=(
                2,
                port,
                self.data_dir,
                self.ckpt_path,
                2,  # numb_steps: train 2 fresh steps from step 0
                False,  # is_restart=False → init_model
                result_dict,
            ),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        from deepmd.dpmodel.utils.learning_rate import (
            LearningRateExp,
        )

        # init_model resets step to 0
        self.assertEqual(r0["start_step"], 0)
        self.assertEqual(r1["start_step"], 0)

        # LR should be lr_schedule(0), i.e. start_lr
        lr_params = self.lr_config.copy()
        lr_params["num_steps"] = 2  # init_model config uses numb_steps=2
        expected_lr = LearningRateExp(**lr_params).value(0)
        self.assertAlmostEqual(r0["init_lr"], expected_lr, places=10)

        # Only rank 0 produces lcurve
        self.assertTrue(r0["lcurve_exists"])
        self.assertFalse(r1["lcurve_exists"])

        # Initial weights (after checkpoint load) must match phase-1 final weights
        for name in self.phase1_weights:
            self.assertIn(name, r0["init_weights"], f"Missing param: {name}")
            torch.testing.assert_close(
                r0["init_weights"][name],
                self.phase1_weights[name],
                msg=f"init_model did not inherit weights correctly: {name}",
            )

        # Initial weights identical across ranks
        for name in r0["init_weights"]:
            torch.testing.assert_close(
                r0["init_weights"][name],
                r1["init_weights"][name],
                msg=f"init_model weights differ across ranks: {name}",
            )


class TestDDPRestart(_DDPResumeBase):
    """DDP restart: continues training from saved step with restored optimizer."""

    def test_ddp_restart(self) -> None:
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_check_resume,
            args=(
                2,
                port,
                self.data_dir,
                self.ckpt_path,
                4,  # numb_steps: continue to step 4
                True,  # is_restart=True → restart_model
                result_dict,
            ),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        from deepmd.dpmodel.utils.learning_rate import (
            LearningRateExp,
        )

        # restart restores the step counter
        self.assertEqual(r0["start_step"], 2)
        self.assertEqual(r1["start_step"], 2)

        # LR should be lr_schedule(2) with num_steps=4 (the restart config)
        lr_params = self.lr_config.copy()
        lr_params["num_steps"] = 4  # restart config uses numb_steps=4
        lr_sched = LearningRateExp(**lr_params)
        expected_lr = lr_sched.value(2)
        start_lr = lr_sched.value(0)
        self.assertAlmostEqual(r0["init_lr"], expected_lr, places=10)
        # Verify it is NOT equal to start_lr (i.e. the LR actually decayed)
        self.assertNotAlmostEqual(
            r0["init_lr"],
            start_lr,
            places=10,
            msg="restart LR should differ from start_lr",
        )

        # Only rank 0 produces lcurve
        self.assertTrue(r0["lcurve_exists"])
        self.assertFalse(r1["lcurve_exists"])

        # Initial weights (after checkpoint load) must match phase-1 final weights
        for name in self.phase1_weights:
            self.assertIn(name, r0["init_weights"], f"Missing param: {name}")
            torch.testing.assert_close(
                r0["init_weights"][name],
                self.phase1_weights[name],
                msg=f"restart did not load weights correctly: {name}",
            )

        # Initial weights identical across ranks
        for name in r0["init_weights"]:
            torch.testing.assert_close(
                r0["init_weights"][name],
                r1["init_weights"][name],
                msg=f"restart weights differ across ranks: {name}",
            )


# ---------------------------------------------------------------------------
# Finetune + DDP tests
# ---------------------------------------------------------------------------


class _DDPFinetuneBase(unittest.TestCase):
    """Shared setup: train pretrained se_e2_a model, save checkpoint + weights."""

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = os.path.join(data_dir, "data_0")

        cls._tmpdir = tempfile.mkdtemp(prefix="ddp_ft_setup_")
        old_cwd = os.getcwd()
        os.chdir(cls._tmpdir)
        try:
            config = _make_config(cls.data_dir, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)
            trainer.run()

            cls.ckpt_path = os.path.join(cls._tmpdir, "model.ckpt.pt")
            assert os.path.exists(cls.ckpt_path), "Pretrained checkpoint not created"

            # Save pretrained state for comparison (excluding _extra_state)
            state = torch.load(cls.ckpt_path, map_location="cpu", weights_only=True)
            model_state = state["model"] if "model" in state else state
            cls.pretrained_state = {
                k: v.clone() for k, v in model_state.items() if k != "_extra_state"
            }
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmpdir, ignore_errors=True)


class TestDDPFinetune(_DDPFinetuneBase):
    """DDP finetune: same type_map, descriptor + fitting from pretrained."""

    def test_ddp_finetune(self) -> None:
        port = _find_free_port()
        result_dict = mp.Manager().dict()

        config = _make_config(self.data_dir, numb_steps=2)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        mp.spawn(
            _worker_finetune,
            args=(2, port, self.ckpt_path, config, None, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # Only rank 0 writes output
        self.assertTrue(r0["lcurve_exists"], "rank 0 should produce lcurve.out")
        self.assertFalse(r1["lcurve_exists"], "rank 1 should NOT produce lcurve.out")
        self.assertGreater(len(r0["ckpt_files"]), 0, "rank 0 should save checkpoints")
        self.assertEqual(len(r1["ckpt_files"]), 0, "rank 1 should NOT save checkpoints")

        # Descriptor + fitting weights must match pretrained
        for key in self.pretrained_state:
            if key in r0["init_state"] and (".descriptor." in key or ".fitting" in key):
                torch.testing.assert_close(
                    r0["init_state"][key],
                    self.pretrained_state[key],
                    msg=f"Weight should match pretrained: {key}",
                )

        # Init state identical across ranks
        for name in r0["init_state"]:
            torch.testing.assert_close(
                r0["init_state"][name],
                r1["init_state"][name],
                msg=f"Finetune init state differs across ranks: {name}",
            )


class TestDDPFinetuneRandomFitting(_DDPFinetuneBase):
    """DDP finetune with random fitting: descriptor from pretrained, fitting random."""

    def test_ddp_finetune_random_fitting(self) -> None:
        port = _find_free_port()
        result_dict = mp.Manager().dict()

        config = _make_config(self.data_dir, numb_steps=2)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        mp.spawn(
            _worker_finetune,
            args=(2, port, self.ckpt_path, config, "RANDOM", result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # Descriptor weights must match pretrained
        for key in self.pretrained_state:
            if key in r0["init_state"] and ".descriptor." in key:
                torch.testing.assert_close(
                    r0["init_state"][key],
                    self.pretrained_state[key],
                    msg=f"Descriptor weight should match pretrained: {key}",
                )

        # Fitting weights should NOT match pretrained (random init)
        # bias_atom_e is set by bias adjustment, not random init — skip it
        has_fitting_diff = False
        for key in self.pretrained_state:
            if (
                key in r0["init_state"]
                and ".fitting" in key
                and "bias_atom_e" not in key
                and r0["init_state"][key].is_floating_point()
            ):
                if not torch.equal(r0["init_state"][key], self.pretrained_state[key]):
                    has_fitting_diff = True
        self.assertTrue(
            has_fitting_diff, "Random fitting should produce different weights"
        )

        # Init state identical across ranks
        for name in r0["init_state"]:
            torch.testing.assert_close(
                r0["init_state"][name],
                r1["init_state"][name],
                msg=f"Finetune random fitting state differs across ranks: {name}",
            )


class TestDDPFinetuneNewType(unittest.TestCase):
    """DDP finetune with type_map change (new type).

    Exercises the ``_unwrapped.model["Default"]`` path (line 712) when
    ``finetune_rule.get_has_new_type()`` is True, plus stat recomputation
    and broadcast for the new type.  Uses DPA1 (mixed_types) which supports
    ``change_type_map``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        raw_data = os.path.join(data_dir, "data_0")

        # Subsample data for faster DPA1 test
        cls._data_tmpdir = tempfile.mkdtemp(prefix="ddp_ft_nt_data_")
        _subsample_data(raw_data, os.path.join(cls._data_tmpdir, "data_0"))
        cls.data_dir = os.path.join(cls._data_tmpdir, "data_0")

        # Train pretrained DPA1 with type_map=["O", "H"]
        cls._train_tmpdir = tempfile.mkdtemp(prefix="ddp_ft_nt_train_")
        old_cwd = os.getcwd()
        os.chdir(cls._train_tmpdir)
        try:
            config = _make_dpa1_config(cls.data_dir, numb_steps=1)
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)
            trainer.run()

            cls.ckpt_path = os.path.join(cls._train_tmpdir, "model.ckpt.pt")
            assert os.path.exists(cls.ckpt_path), (
                "DPA1 pretrained checkpoint not created"
            )
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._data_tmpdir, ignore_errors=True)
        shutil.rmtree(cls._train_tmpdir, ignore_errors=True)

    def test_ddp_finetune_new_type(self) -> None:
        """Finetune DPA1 from ["O","H"] to ["O","H","B"] under DDP."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()

        # Finetune config with new type "B" added
        config = _make_dpa1_config(self.data_dir, numb_steps=2)
        config["model"]["type_map"] = ["O", "H", "B"]
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        mp.spawn(
            _worker_finetune,
            args=(2, port, self.ckpt_path, config, None, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # Training completes without error
        self.assertTrue(r0["lcurve_exists"], "rank 0 should produce lcurve.out")
        self.assertFalse(r1["lcurve_exists"], "rank 1 should NOT produce lcurve.out")

        # Init state identical across ranks (stat broadcast worked)
        for name in r0["init_state"]:
            torch.testing.assert_close(
                r0["init_state"][name],
                r1["init_state"][name],
                msg=f"Finetune new_type init state differs across ranks: {name}",
            )


def _make_dpa1_multitask_config(
    data_dir: str, numb_steps: int = 2, type_map: list | None = None
) -> dict:
    """Build a minimal multi-task DPA1 config (mixed_types) for finetune tests."""
    if type_map is None:
        type_map = ["O", "H"]
    descriptor = {
        "type": "dpa1",
        "sel": 12,
        "rcut_smth": 0.50,
        "rcut": 3.00,
        "neuron": [4, 8],
        "axis_neuron": 4,
        "attn": 4,
        "attn_layer": 1,
        "attn_dotr": True,
        "seed": 1,
    }
    fitting = {
        "neuron": [8, 8],
        "resnet_dt": True,
        "seed": 1,
    }
    return {
        "model": {
            "shared_dict": {
                "my_type_map": list(type_map),
                "my_descriptor": deepcopy(descriptor),
            },
            "model_dict": {
                "model_1": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": deepcopy(fitting),
                    "data_stat_nbatch": 1,
                },
                "model_2": {
                    "type_map": "my_type_map",
                    "descriptor": "my_descriptor",
                    "fitting_net": deepcopy(fitting),
                    "data_stat_nbatch": 1,
                },
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss_dict": {
            "model_1": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "model_2": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
        },
        "training": {
            "model_prob": {
                "model_1": 0.5,
                "model_2": 0.5,
            },
            "data_dict": {
                "model_1": {
                    "stat_file": "./stat_files/model_1",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
                "model_2": {
                    "stat_file": "./stat_files/model_2",
                    "training_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                    },
                    "validation_data": {
                        "systems": [data_dir],
                        "batch_size": 1,
                        "numb_btch": 1,
                    },
                },
            },
            "numb_steps": numb_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": numb_steps,
        },
    }


def _worker_multitask_finetune(
    rank, world_size, port, data_dir, ckpt_path, finetune_config, result_dict
):
    """Worker: DDP multi-task finetune from checkpoint."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_mt_ft_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = deepcopy(finetune_config)
            config["model"], shared_links = preprocess_shared_params(config["model"])
            config = update_deepmd_input(config, warning=False)
            config = normalize(config, multi_task=True)
            config["model"], finetune_links = get_finetune_rules(
                ckpt_path, config["model"]
            )
            trainer = get_trainer(
                config,
                finetune_model=ckpt_path,
                finetune_links=finetune_links,
                shared_links=shared_links,
            )
            # Capture init state before training
            init_state = {
                k: v.detach().cpu().clone()
                for k, v in trainer._unwrapped.state_dict().items()
                if k != "_extra_state"
            }
            trainer.run()
            result_dict[rank] = {
                "init_state": init_state,
                "lcurve_exists": os.path.exists("lcurve.out"),
                "ckpt_files": [f for f in os.listdir(".") if f.endswith(".pt")],
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_single_task_compile_train(rank, world_size, port, data_dir, result_dict):
    """Worker: run single-task DDP training with torch.compile enabled.

    This exercises the ``_compile_model`` code path under DDP, which must
    unwrap ``DistributedDataParallel`` to access ``wrapper.module.model``.
    Before the fix, ``self.wrapper.model[task_key]`` raised ``AttributeError``
    because ``DistributedDataParallel`` does not expose ``.model`` directly.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_compile_st_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_config(data_dir, numb_steps=2)
            config["training"]["enable_compile"] = True
            config = update_deepmd_input(config, warning=False)
            config = normalize(config)
            trainer = get_trainer(config)
            trainer.run()

            from deepmd.pt_expt.train.training import (
                _CompiledModel,
            )

            # Check the compiled model is a _CompiledModel
            is_compiled = isinstance(
                trainer._unwrapped.model["Default"], _CompiledModel
            )

            lcurve_exists = os.path.exists("lcurve.out")
            ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]

            weights = {
                name: p.detach().cpu().clone()
                for name, p in trainer._unwrapped.named_parameters()
            }

            result_dict[rank] = {
                "lcurve_exists": lcurve_exists,
                "num_ckpts": len(ckpt_files),
                "weights": weights,
                "is_compiled": is_compiled,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


def _worker_multitask_compile_train(rank, world_size, port, data_dir, result_dict):
    """Worker: run multi-task DDP training with torch.compile enabled.

    Exercises the per-branch compilation loop in ``_compile_model`` under DDP.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=_DDP_BACKEND, rank=rank, world_size=world_size)
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"ddp_compile_mt_rank{rank}_")
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            config = _make_multitask_config(data_dir, numb_steps=2)
            config["training"]["enable_compile"] = True
            config["model"], shared_links = preprocess_shared_params(config["model"])
            config = update_deepmd_input(config, warning=False)
            config = normalize(config, multi_task=True)
            trainer = get_trainer(config, shared_links=shared_links)
            trainer.run()

            from deepmd.pt_expt.train.training import (
                _CompiledModel,
            )

            # Check both branch models are compiled
            compiled_flags = {}
            for mk in ("model_1", "model_2"):
                compiled_flags[mk] = isinstance(
                    trainer._unwrapped.model[mk], _CompiledModel
                )

            lcurve_exists = os.path.exists("lcurve.out")
            ckpt_files = [f for f in os.listdir(".") if f.endswith(".pt")]

            # Get shared descriptor params from model_1
            desc_params = {}
            for name, p in trainer._unwrapped.model[
                "model_1"
            ].original_model.atomic_model.descriptor.named_parameters():
                desc_params[name] = p.detach().cpu().clone()

            result_dict[rank] = {
                "lcurve_exists": lcurve_exists,
                "num_ckpts": len(ckpt_files),
                "desc_params": desc_params,
                "compiled_flags": compiled_flags,
            }
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        dist.destroy_process_group()


class TestDDPCompileSingleTask(unittest.TestCase):
    """DDP + torch.compile: single-task training with 2 ranks.

    Exercises ``_compile_model`` under DDP, which requires unwrapping
    ``DistributedDataParallel`` to access ``wrapper.module.model``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        cls.data_dir = os.path.join(data_dir, "data_0")

    def test_ddp_compile_single_task(self) -> None:
        """2 ranks, se_e2_a, enable_compile=True, 2 steps."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_single_task_compile_train,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)

        # Both ranks have compiled models
        self.assertTrue(results[0]["is_compiled"], "rank 0 model should be compiled")
        self.assertTrue(results[1]["is_compiled"], "rank 1 model should be compiled")

        # Only rank 0 produces output files
        self.assertTrue(results[0]["lcurve_exists"])
        self.assertFalse(results[1]["lcurve_exists"])
        self.assertGreater(results[0]["num_ckpts"], 0)
        self.assertEqual(results[1]["num_ckpts"], 0)

        # Final weights identical across ranks
        for name in results[0]["weights"]:
            torch.testing.assert_close(
                results[0]["weights"][name],
                results[1]["weights"][name],
                msg=f"Compiled DDP weights differ across ranks: {name}",
            )


class TestDDPCompileMultiTask(unittest.TestCase):
    """DDP + torch.compile: multi-task training with 2 ranks.

    Exercises the per-branch compilation loop in ``_compile_model`` under DDP.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(_PT_DATA):
            raise unittest.SkipTest(f"Test data not found: {_PT_DATA}")
        cls.data_dir = _PT_DATA

    def test_ddp_compile_multitask(self) -> None:
        """2 ranks, multi-task, enable_compile=True, 2 steps."""
        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_multitask_compile_train,
            args=(2, port, self.data_dir, result_dict),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)

        # Both ranks have compiled models for both branches
        for mk in ("model_1", "model_2"):
            self.assertTrue(
                results[0]["compiled_flags"][mk],
                f"rank 0 {mk} should be compiled",
            )
            self.assertTrue(
                results[1]["compiled_flags"][mk],
                f"rank 1 {mk} should be compiled",
            )

        # Only rank 0 produces output files
        self.assertTrue(results[0]["lcurve_exists"])
        self.assertFalse(results[1]["lcurve_exists"])
        self.assertGreater(results[0]["num_ckpts"], 0)
        self.assertEqual(results[1]["num_ckpts"], 0)

        # Shared descriptor params identical across ranks
        for name in results[0]["desc_params"]:
            torch.testing.assert_close(
                results[0]["desc_params"][name],
                results[1]["desc_params"][name],
                msg=f"Compiled DDP shared descriptor param differs: {name}",
            )


class TestDDPMultiTaskFinetune(unittest.TestCase):
    """DDP multi-task finetune with type_map change (new type).

    Trains a 2-branch multi-task DPA1 model with type_map ["O","H"], then
    finetunes 4 branches with extended type_map ["O","H","B"] under DDP.
    Builds a reference state_dict by manually replicating the trainer's
    finetune operations (load pretrained, change_type_map with computed
    model_with_new_type_stat, weight copy) to verify correctness.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from deepmd.pt_expt.model import (
            get_model,
        )
        from deepmd.pt_expt.train.wrapper import (
            ModelWrapper,
        )
        from deepmd.pt_expt.utils.env import (
            DEVICE,
        )
        from deepmd.pt_expt.utils.stat import (
            make_stat_input,
        )
        from deepmd.utils.data import (
            DataRequirementItem,
        )
        from deepmd.utils.data_system import (
            DeepmdDataSystem,
            process_systems,
        )

        data_dir = os.path.join(EXAMPLE_DIR, "data")
        if not os.path.isdir(data_dir):
            raise unittest.SkipTest(f"Example data not found: {data_dir}")
        raw_data = os.path.join(data_dir, "data_0")

        # Subsample data for faster test
        cls._data_tmpdir = tempfile.mkdtemp(prefix="ddp_mt_ft_data_")
        _subsample_data(raw_data, os.path.join(cls._data_tmpdir, "data_0"))
        cls.data_dir = os.path.join(cls._data_tmpdir, "data_0")

        ft_type_map = ["O", "H", "B"]

        # Train pretrained 2-branch multi-task DPA1 model
        cls._train_tmpdir = tempfile.mkdtemp(prefix="ddp_mt_ft_train_")
        old_cwd = os.getcwd()
        os.chdir(cls._train_tmpdir)
        try:
            config = _make_dpa1_multitask_config(cls.data_dir, numb_steps=2)
            config["model"], shared_links = preprocess_shared_params(config["model"])
            config = update_deepmd_input(config, warning=False)
            config = normalize(config, multi_task=True)
            trainer = get_trainer(config, shared_links=shared_links)
            trainer.run()

            cls.ckpt_path = os.path.join(cls._train_tmpdir, "model.ckpt.pt")
            assert os.path.exists(cls.ckpt_path), (
                "DPA1 multi-task pretrained checkpoint not created"
            )

            # Build reference state_dict with extended type_map
            state_dict_full = torch.load(
                cls.ckpt_path, map_location=DEVICE, weights_only=True
            )
            state_dict_ckpt = (
                state_dict_full["model"]
                if "model" in state_dict_full
                else state_dict_full
            )
            pretrained_model_params = state_dict_ckpt["_extra_state"]["model_params"]

            pretrained_models = {}
            for pk in pretrained_model_params["model_dict"]:
                pretrained_models[pk] = get_model(
                    deepcopy(pretrained_model_params["model_dict"][pk])
                ).to(DEVICE)
            pretrained_wrapper = ModelWrapper(pretrained_models)
            pretrained_wrapper.load_state_dict(state_dict_ckpt)

            # Record pretrained state BEFORE change_type_map for O/H
            # inheritance verification
            cls.pretrained_oh_state = {
                k: v.cpu().clone()
                for k, v in pretrained_wrapper.model.state_dict().items()
            }

            # Build model_with_new_type_stat with computed stats
            ref_model_params = deepcopy(
                pretrained_model_params["model_dict"]["model_1"]
            )
            ref_model_params["type_map"] = ft_type_map
            ref_model = get_model(ref_model_params).to(DEVICE)

            energy_data_req = [
                DataRequirementItem(
                    "energy", ndof=1, atomic=False, must=False, high_prec=True
                ),
                DataRequirementItem(
                    "force", ndof=3, atomic=True, must=False, high_prec=False
                ),
                DataRequirementItem(
                    "virial", ndof=9, atomic=False, must=False, high_prec=False
                ),
            ]
            data_systems = process_systems([cls.data_dir])
            data = DeepmdDataSystem(
                systems=data_systems,
                batch_size=1,
                test_size=1,
                type_map=ft_type_map,
                trn_all_set=True,
            )
            data.add_data_requirements(energy_data_req)
            ref_model.compute_or_load_stat(
                sampled_func=lambda: make_stat_input(data, 1),
                stat_file_path=None,
            )

            for pk in pretrained_model_params["model_dict"]:
                pretrained_wrapper.model[pk].change_type_map(
                    ft_type_map,
                    model_with_new_type_stat=ref_model,
                )

            cls.ref_state_dict = pretrained_wrapper.model.state_dict()
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._data_tmpdir, ignore_errors=True)
        shutil.rmtree(cls._train_tmpdir, ignore_errors=True)

    def test_ddp_multitask_finetune(self) -> None:
        """Finetune 4-branch DPA1 from 2-branch with extended type_map under DDP."""
        ft_type_map = ["O", "H", "B"]
        ft_config = _make_dpa1_multitask_config(
            self.data_dir, numb_steps=1, type_map=ft_type_map
        )

        # Add model_3 and model_4
        ft_config["model"]["model_dict"]["model_3"] = deepcopy(
            ft_config["model"]["model_dict"]["model_2"]
        )
        ft_config["model"]["model_dict"]["model_4"] = deepcopy(
            ft_config["model"]["model_dict"]["model_2"]
        )
        ft_config["loss_dict"]["model_3"] = deepcopy(ft_config["loss_dict"]["model_2"])
        ft_config["loss_dict"]["model_4"] = deepcopy(ft_config["loss_dict"]["model_2"])
        ft_config["training"]["model_prob"]["model_3"] = 0.25
        ft_config["training"]["model_prob"]["model_4"] = 0.25
        ft_config["training"]["model_prob"]["model_1"] = 0.25
        ft_config["training"]["model_prob"]["model_2"] = 0.25
        ft_config["training"]["data_dict"]["model_3"] = deepcopy(
            ft_config["training"]["data_dict"]["model_2"]
        )
        ft_config["training"]["data_dict"]["model_3"]["stat_file"] = (
            "./stat_files/model_3"
        )
        ft_config["training"]["data_dict"]["model_4"] = deepcopy(
            ft_config["training"]["data_dict"]["model_2"]
        )
        ft_config["training"]["data_dict"]["model_4"]["stat_file"] = (
            "./stat_files/model_4"
        )

        # Finetune rules:
        # model_1: no finetune_head → resume
        # model_2: finetune from model_2
        ft_config["model"]["model_dict"]["model_2"]["finetune_head"] = "model_2"
        # model_3: finetune from model_2
        ft_config["model"]["model_dict"]["model_3"]["finetune_head"] = "model_2"
        # model_4: no finetune_head, new key → random fitting

        port = _find_free_port()
        result_dict = mp.Manager().dict()
        mp.spawn(
            _worker_multitask_finetune,
            args=(
                2,
                port,
                self.data_dir,
                self.ckpt_path,
                ft_config,
                result_dict,
            ),
            nprocs=2,
            join=True,
        )
        results = dict(result_dict)
        r0, r1 = results[0], results[1]

        # Only rank 0 writes output
        self.assertTrue(r0["lcurve_exists"], "rank 0 should produce lcurve.out")
        self.assertFalse(r1["lcurve_exists"], "rank 1 should NOT produce lcurve.out")

        # Init state identical across ranks (DDP sync for finetune)
        for name in r0["init_state"]:
            torch.testing.assert_close(
                r0["init_state"][name],
                r1["init_state"][name],
                msg=f"Multi-task finetune init state differs across ranks: {name}",
            )

        # Verify weight inheritance against reference (with extended type_map)
        # Keys in init_state have "model." prefix from wrapper; ref_state_dict
        # is from wrapper.model.state_dict() so keys don't have "model." prefix
        ref = self.ref_state_dict
        init = r0["init_state"]
        for key in init:
            # Skip type_embedding (random init for new type B differs)
            if "type_embedding" in key:
                continue
            # Strip "model." prefix
            model_key = key.split("model.", 1)[-1] if key.startswith("model.") else key
            if "model_1" in key:
                # model_1: resume — ALL weights match reference
                if model_key in ref:
                    torch.testing.assert_close(
                        ref[model_key],
                        init[key],
                        msg=f"model_1 (resume) DDP mismatch: {key}",
                    )
            elif "model_2" in key and "out_bias" not in key and "out_std" not in key:
                if model_key in ref:
                    torch.testing.assert_close(
                        ref[model_key],
                        init[key],
                        msg=f"model_2 (finetune) DDP mismatch: {key}",
                    )
            elif "model_3" in key and "out_bias" not in key and "out_std" not in key:
                ref_key = model_key.replace("model_3", "model_2")
                if ref_key in ref:
                    torch.testing.assert_close(
                        ref[ref_key],
                        init[key],
                        msg=f"model_3 (from model_2) DDP mismatch: {key}",
                    )
            elif (
                "model_4" in key
                and "fitting_net" not in key
                and "out_bias" not in key
                and "out_std" not in key
            ):
                ref_key = model_key.replace("model_4", "model_2")
                if ref_key in ref:
                    torch.testing.assert_close(
                        ref[ref_key],
                        init[key],
                        msg=f"model_4 (random) descriptor DDP mismatch: {key}",
                    )

        # Verify O/H descriptor stats are inherited from pretrained (not
        # recomputed).  pretrained_oh_state has shape [2,...] for O,H;
        # finetuned init has shape [3,...] for O,H,B.
        _STAT_SUFFIXES = ("mean", "stddev", "davg", "dstd")
        n_old = 2  # ["O", "H"]
        n_new = 3  # ["O", "H", "B"]
        checked_count = 0
        pretrained_oh = self.pretrained_oh_state
        for key in init:
            if "type_embedding" in key:
                continue
            if not any(key.endswith(s) for s in _STAT_SUFFIXES):
                continue
            # Use model_1 (all branches share descriptor after share_params)
            if "model_1" not in key:
                continue
            # init_state has "model." prefix; pretrained_oh_state doesn't
            pre_key = key.split("model.", 1)[-1] if key.startswith("model.") else key
            if pre_key not in pretrained_oh:
                continue
            pre_val = pretrained_oh[pre_key]
            ft_val = init[key]
            # Find the type axis (size grew from n_old to n_new)
            for ax in range(pre_val.ndim):
                if pre_val.shape[ax] == n_old and ft_val.shape[ax] == n_new:
                    for ti, tname in enumerate(["O", "H"]):
                        torch.testing.assert_close(
                            ft_val.select(ax, ti),
                            pre_val.select(ax, ti),
                            msg=(f"{tname} stat not inherited from pretrained: {key}"),
                        )
                    checked_count += 1
                    break
        self.assertGreater(
            checked_count,
            0,
            "No descriptor stat keys found for O/H inheritance check",
        )


if __name__ == "__main__":
    unittest.main()
