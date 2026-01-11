# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    ClassVar,
)

import deepmd
from deepmd.env import (
    GLOBAL_CONFIG,
    get_default_nthreads,
    global_float_prec,
)
from deepmd.utils.hostlist import (
    get_host_names,
)

log = logging.getLogger(__name__)


class SummaryPrinter(ABC):
    """Base summary printer.

    Backends should inherit from this class and implement the abstract methods.
    """

    # http://patorjk.com/software/taag. Font:Big"
    WELCOME = (
        r" _____               _____   __  __  _____           _     _  _   ",
        r"|  __ \             |  __ \ |  \/  ||  __ \         | |   (_)| |  ",
        r"| |  | |  ___   ___ | |__) || \  / || |  | | ______ | | __ _ | |_ ",
        r"| |  | | / _ \ / _ \|  ___/ | |\/| || |  | ||______|| |/ /| || __|",
        r"| |__| ||  __/|  __/| |     | |  | || |__| |        |   < | || |_ ",
        r"|_____/  \___| \___||_|     |_|  |_||_____/         |_|\_\|_| \__|",
    )

    CITATION = (
        "Please read and cite:",
        "Wang, Zhang, Han and E, Comput.Phys.Comm. 228, 178-184 (2018)",
        "Zeng et al, J. Chem. Phys., 159, 054801 (2023)",
        "Zeng et al, J. Chem. Theory Comput., 21, 4375-4385 (2025)",
        "See https://deepmd.rtfd.io/credits/ for details.",
    )

    BUILD: ClassVar = {
        "Installed To": "\n".join(deepmd.__path__),
        "Source": GLOBAL_CONFIG["git_summ"],
        "Source Branch": GLOBAL_CONFIG["git_branch"],
        "Source Commit": GLOBAL_CONFIG["git_hash"],
        "Source Commit At": GLOBAL_CONFIG["git_date"],
        "Float Precision": global_float_prec.capitalize(),
        "Build Variant": GLOBAL_CONFIG["dp_variant"].upper(),
    }

    def __call__(self) -> None:
        """Print build and current running cluster configuration summary."""
        nodename, nodelist = get_host_names()
        build_info = self.BUILD.copy()
        build_info.update(self.get_backend_info())
        if len(nodelist) > 1:
            build_info.update(
                {
                    "World Size": str(len(nodelist)),
                    "Node List": ", ".join(set(nodelist)),
                }
            )
        build_info.update(
            {
                "Running On": nodename,
                "Computing Device": self.get_compute_device().upper(),
            }
        )
        device_name = self.get_device_name()
        if device_name:
            build_info["Device Name"] = device_name
        if self.is_built_with_cuda():
            env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
            build_info["CUDA_VISIBLE_DEVICES"] = env_value
        if self.is_built_with_rocm():
            env_value = os.environ.get("HIP_VISIBLE_DEVICES", "unset")
            build_info["HIP_VISIBLE_DEVICES"] = env_value
        if self.is_built_with_cuda() or self.is_built_with_rocm():
            build_info["Visible GPU Count"] = str(self.get_ngpus())

        intra, inter = get_default_nthreads()
        build_info.update(
            {
                "Num Intra Threads": str(intra),
                "Num Inter Threads": str(inter),
            }
        )
        # count the maximum characters in the keys and values
        max_key_len = max(len(k) for k in build_info) + 2
        max_val_len = max(
            len(x) for v in build_info.values() for x in str(v).split("\n")
        )
        # print the summary
        for line in self.WELCOME + self.CITATION:
            log.info(line)
        log.info("-" * (max_key_len + max_val_len))
        for kk, vv in build_info.items():
            for iline, vline in enumerate(str(vv).split("\n")):
                if iline == 0:
                    log.info(f"{kk + ': ':<{max_key_len}}{vline}")
                else:
                    log.info(f"{'':<{max_key_len}}{vline}")
        log.info("-" * (max_key_len + max_val_len))

    @abstractmethod
    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""

    @abstractmethod
    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""

    @abstractmethod
    def get_compute_device(self) -> str:
        """Get Compute device."""

    @abstractmethod
    def get_ngpus(self) -> int:
        """Get the number of GPUs."""

    @abstractmethod
    def get_device_name(self) -> str | None:
        """Get the device name (e.g., NVIDIA A800-SXM4-80GB) if available."""

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {}
