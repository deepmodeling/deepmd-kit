# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging
import math
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
)

from deepmd.common import (
    expand_sys_str,
)
from deepmd.pt.loss import (
    DenoiseLoss,
    EnergyStdLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
    JIT,
    NUM_WORKERS,
)

if torch.__version__.startswith("2"):
    import torch._dynamo
log = logging.getLogger(__name__)


class Tester:
    def __init__(
        self,
        model_ckpt,
        input_script=None,
        system=None,
        datafile=None,
        numb_test=100,
        detail_file=None,
        shuffle_test=False,
        head=None,
    ):
        """Construct a DeePMD tester.

        Args:
        - config: The Dict-like configuration with training options.
        """
        self.numb_test = numb_test
        self.detail_file = detail_file
        self.shuffle_test = shuffle_test
        # Model
        state_dict = torch.load(model_ckpt, map_location=DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = state_dict["_extra_state"]["model_params"]
        self.multi_task = "model_dict" in model_params
        if self.multi_task:
            assert head is not None, "Head must be specified in multitask mode!"
            self.head = head
            assert head in model_params["model_dict"], (
                f"Specified head {head} not found in model {model_ckpt}! "
                f"Available ones are {list(model_params['model_dict'].keys())}."
            )
            model_params = model_params["model_dict"][head]
            state_dict_head = {"_extra_state": state_dict["_extra_state"]}
            for item in state_dict:
                if f"model.{head}." in item:
                    state_dict_head[
                        item.replace(f"model.{head}.", "model.Default.")
                    ] = state_dict[item].clone()
            state_dict = state_dict_head

        # Data
        if input_script is not None:
            with open(input_script) as fin:
                self.input_script = json.load(fin)
            training_params = self.input_script["training"]
            if not self.multi_task:
                assert (
                    "validation_data" in training_params
                ), f"Validation systems not found in {input_script}!"
                self.systems = training_params["validation_data"]["systems"]
                self.batchsize = training_params["validation_data"]["batch_size"]
                log.info(f"Testing validation systems in input script: {input_script}")
            else:
                assert (
                    "data_dict" in training_params
                ), f"Input script {input_script} is not in multi-task mode!"
                assert head in training_params["data_dict"], (
                    f"Specified head {head} not found in input script {input_script}! "
                    f"Available ones are {list(training_params['data_dict'].keys())}."
                )
                assert (
                    "validation_data" in training_params["data_dict"][head]
                ), f"Validation systems not found in head {head} of {input_script}!"
                self.systems = training_params["data_dict"][head]["validation_data"][
                    "systems"
                ]
                self.batchsize = training_params["data_dict"][head]["validation_data"][
                    "batch_size"
                ]
                log.info(
                    f"Testing validation systems in head {head} of input script: {input_script}"
                )
        elif system is not None:
            self.systems = expand_sys_str(system)
            self.batchsize = "auto"
            log.info("Testing systems in path: %s", system)
        elif datafile is not None:
            with open(datafile) as fin:
                self.systems = fin.read().splitlines()
            self.batchsize = "auto"
            log.info("Testing systems in file: %s", datafile)
        else:
            self.systems = None
            self.batchsize = None

        self.type_split = False
        if model_params["descriptor"]["type"] in ["se_e2_a"]:
            self.type_split = True
        self.model_params = deepcopy(model_params)
        model_params["resuming"] = True
        self.model = get_model(model_params).to(DEVICE)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model)  # inference only
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)
        self.wrapper.load_state_dict(state_dict)

        # Loss
        if "fitting_net" not in model_params:
            assert (
                input_script is not None
            ), "Denoise model must use --input-script mode!"
            loss_params = self.input_script["loss"]
            loss_type = loss_params.pop("type", "ener")
            assert (
                loss_type == "denoise"
            ), "Models without fitting_net only support denoise test!"
            self.noise_settings = {
                "noise_type": loss_params.pop("noise_type", "uniform"),
                "noise": loss_params.pop("noise", 1.0),
                "noise_mode": loss_params.pop("noise_mode", "fix_num"),
                "mask_num": loss_params.pop("mask_num", 8),
                "same_mask": loss_params.pop("same_mask", False),
                "mask_coord": loss_params.pop("mask_coord", False),
                "mask_type": loss_params.pop("mask_type", False),
                "mask_type_idx": len(model_params["type_map"]) - 1,
            }
            loss_params["ntypes"] = len(model_params["type_map"])
            self.loss = DenoiseLoss(**loss_params)
        else:
            self.noise_settings = None
            self.loss = EnergyStdLoss(inference=True)

    @staticmethod
    def get_data(data):
        with torch.device("cpu"):
            batch_data = next(iter(data))
        for key in batch_data.keys():
            if key == "sid" or key == "fid":
                continue
            elif not isinstance(batch_data[key], list):
                if batch_data[key] is not None:
                    batch_data[key] = batch_data[key].to(DEVICE)
            else:
                batch_data[key] = [item.to(DEVICE) for item in batch_data[key]]
        input_dict = {}
        for item in [
            "coord",
            "atype",
            "box",
        ]:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in [
            "energy",
            "force",
            "virial",
            "clean_coord",
            "clean_type",
            "coord_mask",
            "type_mask",
        ]:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        return input_dict, label_dict

    def run(self):
        systems = self.systems
        system_results = {}
        global_sum_natoms = 0
        for cc, system in enumerate(systems):
            log.info("# ---------------output of dp test--------------- ")
            log.info(f"# testing system : {system}")
            system_pred = []
            system_label = []
            dataset = DpLoaderSet(
                [system],
                self.batchsize,
                self.model_params,
                shuffle=self.shuffle_test,
            )
            sampler = RandomSampler(
                dataset, replacement=True, num_samples=dataset.total_batch
            )
            if sampler is None:
                log.warning(
                    "Sampler not specified!"
                )  # None sampler will lead to a premature stop iteration. Replacement should be True in attribute of the sampler to produce expected number of items in one iteration.
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=None,
                num_workers=min(
                    NUM_WORKERS, 1
                ),  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
            )
            with torch.device("cpu"):
                data = iter(dataloader)

            single_results = {}
            sum_natoms = 0
            sys_natoms = None
            for ii in range(self.numb_test):
                try:
                    input_dict, label_dict = self.get_data(data)
                except StopIteration:
                    if (
                        ii < dataset.total_batch
                    ):  # Unexpected stop iteration.(test step < total batch)
                        raise StopIteration
                    else:
                        break
                model_pred, _, _ = self.wrapper(**input_dict)
                system_pred.append(
                    {
                        item: model_pred[item].detach().cpu().numpy()
                        for item in model_pred
                    }
                )
                system_label.append(
                    {
                        item: label_dict[item].detach().cpu().numpy()
                        for item in label_dict
                    }
                )
                natoms = int(input_dict["atype"].shape[-1])
                _, more_loss = self.loss(
                    model_pred, label_dict, natoms, 1.0, mae=True
                )  # TODO: lr here is useless
                if sys_natoms is None:
                    sys_natoms = natoms
                else:
                    assert (
                        sys_natoms == natoms
                    ), "Frames in one system must be the same!"
                sum_natoms += natoms
                for k, v in more_loss.items():
                    if "mae" in k:
                        single_results[k] = single_results.get(k, 0.0) + v * natoms
                    else:
                        single_results[k] = single_results.get(k, 0.0) + v**2 * natoms
            if self.detail_file is not None:
                save_detail_file(
                    Path(self.detail_file),
                    system_pred,
                    system_label,
                    sys_natoms,
                    system_name=system,
                    append=(cc != 0),
                )
            results = {
                k: v / sum_natoms if "mae" in k else math.sqrt(v / sum_natoms)
                for k, v in single_results.items()
            }
            for item in sorted(results.keys()):
                log.info(f"{item}: {results[item]:.4f}")
            log.info("# ----------------------------------------------- ")
            for k, v in single_results.items():
                system_results[k] = system_results.get(k, 0.0) + v
            global_sum_natoms += sum_natoms

        global_results = {
            k: v / global_sum_natoms if "mae" in k else math.sqrt(v / global_sum_natoms)
            for k, v in system_results.items()
        }
        log.info("# ----------weighted average of errors----------- ")
        if not self.multi_task:
            log.info(f"# number of systems : {len(systems)}")
        else:
            log.info(f"# number of systems for {self.head}: {len(systems)}")
        for item in sorted(global_results.keys()):
            log.info(f"{item}: {global_results[item]:.4f}")
        log.info("# ----------------------------------------------- ")
        return global_results


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
):
    """Save numpy array to test file.

    Parameters
    ----------
    fname : str
        filename
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended insted of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)


def save_detail_file(
    detail_path, system_pred, system_label, natoms, system_name, append=False
):
    ntest = len(system_pred)
    data_e = np.concatenate([item["energy"] for item in system_label]).reshape([-1, 1])
    pred_e = np.concatenate([item["energy"] for item in system_pred]).reshape([-1, 1])
    pe = np.concatenate(
        (
            data_e,
            pred_e,
        ),
        axis=1,
    )
    save_txt_file(
        detail_path.with_suffix(".e.out"),
        pe,
        header="%s: data_e pred_e" % system_name,
        append=append,
    )
    pe_atom = pe / natoms
    save_txt_file(
        detail_path.with_suffix(".e_peratom.out"),
        pe_atom,
        header="%s: data_e pred_e" % system_name,
        append=append,
    )
    if "force" in system_pred[0]:
        data_f = np.concatenate([item["force"] for item in system_label]).reshape(
            [-1, 3]
        )
        pred_f = np.concatenate([item["force"] for item in system_pred]).reshape(
            [-1, 3]
        )
        pf = np.concatenate(
            (
                data_f,
                pred_f,
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".f.out"),
            pf,
            header="%s: data_fx data_fy data_fz pred_fx pred_fy pred_fz" % system_name,
            append=append,
        )
    if "virial" in system_pred[0]:
        data_v = np.concatenate([item["virial"] for item in system_label]).reshape(
            [-1, 9]
        )
        pred_v = np.concatenate([item["virial"] for item in system_pred]).reshape(
            [-1, 9]
        )
        pv = np.concatenate(
            (
                data_v,
                pred_v,
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".v.out"),
            pv,
            header=f"{system_name}: data_vxx data_vxy data_vxz data_vyx data_vyy "
            "data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx "
            "pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz",
            append=append,
        )
        pv_atom = pv / natoms
        save_txt_file(
            detail_path.with_suffix(".v_peratom.out"),
            pv_atom,
            header=f"{system_name}: data_vxx data_vxy data_vxz data_vyx data_vyy "
            "data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx "
            "pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz",
            append=append,
        )
