# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
)

import numpy as np
import torch
from deepmd.model_format import (
    FittingOutputDef,
)

from deepmd.pt.model.task.task import (
    TaskBaseMethod,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.plugin import (
    Plugin,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)


class Fitting(TaskBaseMethod):
    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable:
        """Register a Fitting plugin.

        Parameters
        ----------
        key : str
            the key of a Fitting

        Returns
        -------
        Fitting
            the registered Fitting

        Examples
        --------
        >>> @Fitting.register("some_fitting")
            class SomeFitting(Fitting):
                pass
        """
        return Fitting.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            try:
                fitting_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of fitting should be set by `type`")
            if fitting_type in Fitting.__plugins.plugins:
                cls = Fitting.__plugins.plugins[fitting_type]
            else:
                raise RuntimeError("Unknown descriptor type: " + fitting_type)
        return super().__new__(cls)

    def output_def(self) -> FittingOutputDef:
        """Definition for the task Output."""
        raise NotImplementedError

    def forward(self, **kwargs):
        """Task Output."""
        raise NotImplementedError

    def share_params(self, base_class, shared_level, resume=False):
        assert (
            self.__class__ == base_class.__class__
        ), "Only fitting nets of the same type can share params!"
        if shared_level == 0:
            # link buffers
            if hasattr(self, "bias_atom_e"):
                self.bias_atom_e = base_class.bias_atom_e
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        elif shared_level == 1:
            # only not share the bias_atom_e
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        elif shared_level == 2:
            # share all the layers before final layer
            # the following will successfully link all the params except buffers, which need manually link.
            self._modules["filter_layers"][0].deep_layers = base_class._modules[
                "filter_layers"
            ][0].deep_layers
        elif shared_level == 3:
            # share the first layers
            # the following will successfully link all the params except buffers, which need manually link.
            self._modules["filter_layers"][0].deep_layers[0] = base_class._modules[
                "filter_layers"
            ][0].deep_layers[0]
        else:
            raise NotImplementedError

    def change_energy_bias(
        self, config, model, old_type_map, new_type_map, bias_shift="delta", ntest=10
    ):
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        config : Dict
            The configuration.
        model : EnergyModel
            Energy model loaded pre-trained model.
        new_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        old_type_map : str
            The full type_map in pretrained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        ntest : int
            The number of test samples in a system to change the energy bias.
        """
        logging.info(
            "Changing energy bias in pretrained model for types {}... "
            "(this step may take long time)".format(str(new_type_map))
        )
        # data
        systems = config["training"]["training_data"]["systems"]
        finetune_data = DpLoaderSet(
            systems, ntest, config["model"], type_split=False, noise_settings=None
        )
        sampled = make_stat_input(finetune_data.systems, finetune_data.dataloaders, 1)
        # map
        sorter = np.argsort(old_type_map)
        idx_type_map = sorter[
            np.searchsorted(old_type_map, new_type_map, sorter=sorter)
        ]
        mixed_type = np.all([i.mixed_type for i in finetune_data.systems])
        numb_type = len(old_type_map)
        type_numbs, energy_ground_truth, energy_predict = [], [], []
        for test_data in sampled:
            nframes = test_data["energy"].shape[0]
            if mixed_type:
                atype = test_data["atype"].detach().cpu().numpy()
            else:
                atype = test_data["atype"][0].detach().cpu().numpy()
            assert np.array(
                [i.item() in idx_type_map for i in list(set(atype.reshape(-1)))]
            ).all(), "Some types are not in 'type_map'!"
            energy_ground_truth.append(test_data["energy"].cpu().numpy())
            if mixed_type:
                type_numbs.append(
                    np.array(
                        [(atype == i).sum(axis=-1) for i in idx_type_map],
                        dtype=np.int32,
                    ).T
                )
            else:
                type_numbs.append(
                    np.tile(
                        np.bincount(atype, minlength=numb_type)[idx_type_map],
                        (nframes, 1),
                    )
                )
            if bias_shift == "delta":
                coord = test_data["coord"].to(DEVICE)
                atype = test_data["atype"].to(DEVICE)
                box = (
                    test_data["box"].to(DEVICE)
                    if test_data["box"] is not None
                    else None
                )
                ret = model(coord, atype, box)
                energy_predict.append(
                    ret["energy"].reshape([nframes, 1]).detach().cpu().numpy()
                )
        type_numbs = np.concatenate(type_numbs)
        energy_ground_truth = np.concatenate(energy_ground_truth)
        old_bias = self.bias_atom_e[idx_type_map]
        if bias_shift == "delta":
            energy_predict = np.concatenate(energy_predict)
            bias_diff = energy_ground_truth - energy_predict
            delta_bias = np.linalg.lstsq(type_numbs, bias_diff, rcond=None)[0]
            unbias_e = energy_predict + type_numbs @ delta_bias
            atom_numbs = type_numbs.sum(-1)
            rmse_ae = np.sqrt(
                np.mean(
                    np.square(
                        (unbias_e.ravel() - energy_ground_truth.ravel()) / atom_numbs
                    )
                )
            )
            self.bias_atom_e[idx_type_map] += torch.from_numpy(
                delta_bias.reshape(-1)
            ).to(DEVICE)
            logging.info(
                f"RMSE of atomic energy after linear regression is: {rmse_ae:10.5e} eV/atom."
            )
        elif bias_shift == "statistic":
            statistic_bias = np.linalg.lstsq(
                type_numbs, energy_ground_truth, rcond=None
            )[0]
            self.bias_atom_e[idx_type_map] = (
                torch.from_numpy(statistic_bias.reshape(-1))
                .type_as(self.bias_atom_e[idx_type_map])
                .to(DEVICE)
            )
        else:
            raise RuntimeError("Unknown bias_shift mode: " + bias_shift)
        logging.info(
            "Change energy bias of {} from {} to {}.".format(
                str(new_type_map),
                str(old_bias.detach().cpu().numpy()),
                str(self.bias_atom_e[idx_type_map].detach().cpu().numpy()),
            )
        )
        return None
