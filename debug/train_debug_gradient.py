#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for locating gradient explosion in SeZM-Net + ZBL training.

This script uses torch.autograd.set_detect_anomaly and gradient hooks to
pinpoint the exact location of NaN/Inf gradients.
"""

from __future__ import (
    annotations,
)

import logging
import os
import pdb
import sys
from pathlib import (
    Path,
)

import torch  # noqa: TID253

# Add the deepmd-kit root to Python path
deepmd_root = Path(__file__).parent.parent
sys.path.insert(0, str(deepmd_root))

# Enable anomaly detection BEFORE importing deepmd modules
torch.autograd.set_detect_anomaly(True)


def register_gradient_hooks(model: torch.nn.Module, log: logging.Logger) -> None:
    """Register hooks to monitor gradients for all parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to monitor.
    log : logging.Logger
        Logger for output.
    """

    def make_hook(name: str) -> callable:
        def hook(grad: torch.Tensor) -> None:
            if grad is None:
                return
            if torch.isnan(grad).any():
                log.error(f"NaN gradient detected in: {name}")
                log.error(f"  Gradient shape: {grad.shape}")
                log.error(f"  Gradient stats: min={grad.min()}, max={grad.max()}")
                # Set a breakpoint here for debugging
                pdb.set_trace()
            elif torch.isinf(grad).any():
                log.error(f"Inf gradient detected in: {name}")
                log.error(f"  Gradient shape: {grad.shape}")
                log.error(f"  Gradient stats: min={grad.min()}, max={grad.max()}")
                pdb.set_trace()
            elif grad.abs().max() > 1e6:
                log.warning(f"Large gradient detected in: {name}")
                log.warning(f"  Gradient max abs: {grad.abs().max()}")

        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))


def register_tensor_hooks(model: torch.nn.Module, log: logging.Logger) -> list:
    """Register forward hooks to monitor intermediate tensors.

    Parameters
    ----------
    model : torch.nn.Module
        The model to monitor.
    log : logging.Logger
        Logger for output.

    Returns
    -------
    list
        List of hook handles for cleanup.
    """
    handles = []

    def make_forward_hook(name: str) -> callable:
        def hook(module: torch.nn.Module, input: tuple, output: object) -> None:
            # Check inputs
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if torch.isnan(inp).any():
                        log.error(f"NaN in input[{i}] of {name}")
                        pdb.set_trace()
                    elif torch.isinf(inp).any():
                        log.error(f"Inf in input[{i}] of {name}")
                        pdb.set_trace()

            # Check outputs
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    log.error(f"NaN in output of {name}")
                    log.error(f"  Output shape: {output.shape}")
                    pdb.set_trace()
                elif torch.isinf(output).any():
                    log.error(f"Inf in output of {name}")
                    log.error(f"  Output shape: {output.shape}")
                    pdb.set_trace()
            elif isinstance(output, tuple):
                for j, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        if torch.isnan(out).any():
                            log.error(f"NaN in output[{j}] of {name}")
                            pdb.set_trace()
                        elif torch.isinf(out).any():
                            log.error(f"Inf in output[{j}] of {name}")
                            pdb.set_trace()

        return hook

    for name, module in model.named_modules():
        h = module.register_forward_hook(make_forward_hook(name))
        handles.append(h)

    return handles


def train_with_debug() -> None:
    """Train with gradient debugging enabled."""
    from deepmd.pt.entrypoints.main import (
        train,
    )
    from deepmd.pt.train.training import (
        Trainer,
    )

    # Patch Trainer to add hooks
    original_init = Trainer.__init__

    def patched_init(self: Trainer, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        log = logging.getLogger("GradientDebug")
        log.info("Registering gradient hooks...")
        register_gradient_hooks(self.wrapper, log)
        # Note: forward hooks can slow down training significantly
        # Uncomment if you need to debug forward pass as well:
        # register_tensor_hooks(self.wrapper, log)

    Trainer.__init__ = patched_init

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Set working directory
    work_dir = Path("/home/outisli/Research/dp_train/se_zm/pair/l_2")
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        log.info(f"Working directory: {work_dir}")
        log.info("Anomaly detection enabled - will show traceback on NaN/Inf")

        train(
            input_file="input.json",
            init_model=None,
            restart=None,
            finetune=None,
            init_frz_model=None,
            model_branch="default",
            skip_neighbor_stat=True,
            use_pretrain_script=False,
            force_load=False,
            compile_model=False,
            output="out.json",
        )
    except RuntimeError as e:
        if "nan" in str(e).lower() or "inf" in str(e).lower():
            log.error(f"Gradient anomaly detected: {e}")
            log.error("The traceback above shows where the NaN/Inf was introduced.")
            pdb.post_mortem()
        else:
            raise
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    train_with_debug()
