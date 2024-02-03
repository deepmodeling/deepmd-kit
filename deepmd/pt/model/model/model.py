# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os

import numpy as np
import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)


class BaseModel(torch.nn.Module):
    def __init__(self):
        """Construct a basic model for different tasks."""
        super().__init__()

    def compute_or_load_stat(
        self,
        fitting_param,
        ntypes,
        resuming=False,
        type_map=None,
        stat_file_dir=None,
        stat_file_path=None,
        sampled=None,
    ):
        if fitting_param is None:
            fitting_param = {}
        if not resuming:
            if sampled is not None:  # compute stat
                for sys in sampled:
                    for key in sys:
                        if isinstance(sys[key], list):
                            sys[key] = [item.to(env.DEVICE) for item in sys[key]]
                        else:
                            if sys[key] is not None:
                                sys[key] = sys[key].to(env.DEVICE)
                sumr, suma, sumn, sumr2, suma2 = self.descriptor.compute_input_stats(
                    sampled
                )

                energy = [item["energy"] for item in sampled]
                mixed_type = "real_natoms_vec" in sampled[0]
                if mixed_type:
                    input_natoms = [item["real_natoms_vec"] for item in sampled]
                else:
                    input_natoms = [item["natoms"] for item in sampled]
                tmp = compute_output_stats(energy, input_natoms)
                fitting_param["bias_atom_e"] = tmp[:, 0]
                if stat_file_path is not None:
                    if not os.path.exists(stat_file_dir):
                        os.mkdir(stat_file_dir)
                    if not isinstance(stat_file_path, list):
                        logging.info(f"Saving stat file to {stat_file_path}")
                        np.savez_compressed(
                            stat_file_path,
                            sumr=sumr,
                            suma=suma,
                            sumn=sumn,
                            sumr2=sumr2,
                            suma2=suma2,
                            bias_atom_e=fitting_param["bias_atom_e"],
                            type_map=type_map,
                        )
                    else:
                        for ii, file_path in enumerate(stat_file_path):
                            logging.info(f"Saving stat file to {file_path}")
                            np.savez_compressed(
                                file_path,
                                sumr=sumr[ii],
                                suma=suma[ii],
                                sumn=sumn[ii],
                                sumr2=sumr2[ii],
                                suma2=suma2[ii],
                                bias_atom_e=fitting_param["bias_atom_e"],
                                type_map=type_map,
                            )
            else:  # load stat
                target_type_map = type_map
                if not isinstance(stat_file_path, list):
                    logging.info(f"Loading stat file from {stat_file_path}")
                    stats = np.load(stat_file_path)
                    stat_type_map = list(stats["type_map"])
                    missing_type = [
                        i for i in target_type_map if i not in stat_type_map
                    ]
                    assert not missing_type, f"These type are not in stat file {stat_file_path}: {missing_type}! Please change the stat file path!"
                    idx_map = [stat_type_map.index(i) for i in target_type_map]
                    if stats["sumr"].size:
                        sumr, suma, sumn, sumr2, suma2 = (
                            stats["sumr"][idx_map],
                            stats["suma"][idx_map],
                            stats["sumn"][idx_map],
                            stats["sumr2"][idx_map],
                            stats["suma2"][idx_map],
                        )
                    else:
                        sumr, suma, sumn, sumr2, suma2 = [], [], [], [], []
                    fitting_param["bias_atom_e"] = stats["bias_atom_e"][idx_map]
                else:
                    sumr, suma, sumn, sumr2, suma2 = [], [], [], [], []
                    id_bias_atom_e = None
                    for ii, file_path in enumerate(stat_file_path):
                        logging.info(f"Loading stat file from {file_path}")
                        stats = np.load(file_path)
                        stat_type_map = list(stats["type_map"])
                        missing_type = [
                            i for i in target_type_map if i not in stat_type_map
                        ]
                        assert not missing_type, f"These type are not in stat file {file_path}: {missing_type}! Please change the stat file path!"
                        idx_map = [stat_type_map.index(i) for i in target_type_map]
                        if stats["sumr"].size:
                            sumr_tmp, suma_tmp, sumn_tmp, sumr2_tmp, suma2_tmp = (
                                stats["sumr"][idx_map],
                                stats["suma"][idx_map],
                                stats["sumn"][idx_map],
                                stats["sumr2"][idx_map],
                                stats["suma2"][idx_map],
                            )
                        else:
                            sumr_tmp, suma_tmp, sumn_tmp, sumr2_tmp, suma2_tmp = (
                                [],
                                [],
                                [],
                                [],
                                [],
                            )
                        sumr.append(sumr_tmp)
                        suma.append(suma_tmp)
                        sumn.append(sumn_tmp)
                        sumr2.append(sumr2_tmp)
                        suma2.append(suma2_tmp)
                        fitting_param["bias_atom_e"] = stats["bias_atom_e"][idx_map]
                        if id_bias_atom_e is None:
                            id_bias_atom_e = fitting_param["bias_atom_e"]
                        else:
                            assert (
                                id_bias_atom_e == fitting_param["bias_atom_e"]
                            ).all(), "bias_atom_e in stat files are not consistent!"
            self.descriptor.init_desc_stat(sumr, suma, sumn, sumr2, suma2)
        else:  # resuming for checkpoint; init model params from scratch
            fitting_param["bias_atom_e"] = [0.0] * ntypes
