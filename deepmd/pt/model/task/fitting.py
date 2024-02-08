# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
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

log = logging.getLogger(__name__)


class Fitting(torch.nn.Module, BaseFitting):
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
                raise RuntimeError("Unknown fitting type: " + fitting_type)
        return super().__new__(cls)

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

    @classmethod
    def get_stat_name(cls, ntypes, type_name="ener", **kwargs):
        """
        Get the name for the statistic file of the fitting.
        Usually use the combination of fitting net name and ntypes as the statistic file name.
        """
        if cls is not Fitting:
            raise NotImplementedError("get_stat_name is not implemented!")
        fitting_type = type_name
        return Fitting.__plugins.plugins[fitting_type].get_stat_name(
            ntypes, type_name, **kwargs
        )

    @property
    def data_stat_key(self):
        """
        Get the keys for the data statistic of the fitting.
        Return a list of statistic names needed, such as "bias_atom_e".
        """
        raise NotImplementedError("data_stat_key is not implemented!")

    def compute_or_load_stat(
        self,
        type_map: List[str],
        sampled=None,
        stat_file_path: Optional[Union[str, List[str]]] = None,
    ):
        """
        Compute or load the statistics parameters of the fitting net.
        Calculate and save the output bias to `stat_file_path`
        if `sampled` is not None, otherwise load them from `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        fitting_stat_key = self.data_stat_key
        if sampled is not None:
            tmp_dict = self.compute_output_stats(sampled)
            result_dict = {key: tmp_dict[key] for key in fitting_stat_key}
            result_dict["type_map"] = type_map
            self.save_stats(result_dict, stat_file_path)
        else:  # load the statistics results
            assert stat_file_path is not None, "No stat file to load!"
            result_dict = self.load_stats(type_map, stat_file_path)
        self.init_fitting_stat(**result_dict)

    def save_stats(self, result_dict, stat_file_path: str):
        """
        Save the statistics results to `stat_file_path`.

        Parameters
        ----------
        result_dict
            The dictionary of statistics results.
        stat_file_path
            The path to the statistics file(s).
        """
        log.info(f"Saving stat file to {stat_file_path}")
        np.savez_compressed(stat_file_path, **result_dict)

    def load_stats(self, type_map, stat_file_path: str):
        """
        Load the statistics results to `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        stat_file_path
            The path to the statistics file(s).

        Returns
        -------
        result_dict
            The dictionary of statistics results.
        """
        fitting_stat_key = self.data_stat_key
        target_type_map = type_map
        log.info(f"Loading stat file from {stat_file_path}")
        stats = np.load(stat_file_path)
        stat_type_map = list(stats["type_map"])
        missing_type = [i for i in target_type_map if i not in stat_type_map]
        assert not missing_type, (
            f"These type are not in stat file {stat_file_path}: {missing_type}! "
            f"Please change the stat file path!"
        )
        idx_map = [stat_type_map.index(i) for i in target_type_map]
        if stats[fitting_stat_key[0]].size:  # not empty
            result_dict = {key: stats[key][idx_map] for key in fitting_stat_key}
        else:
            result_dict = {key: [] for key in fitting_stat_key}
        return result_dict

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
        log.info(
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
            log.info(
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
        log.info(
            "Change energy bias of {} from {} to {}.".format(
                str(new_type_map),
                str(old_bias.detach().cpu().numpy()),
                str(self.bias_atom_e[idx_type_map].detach().cpu().numpy()),
            )
        )
        return None
