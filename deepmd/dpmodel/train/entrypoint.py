# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent training entrypoint pipeline."""

from __future__ import (
    annotations,
)

import json
import logging
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

from deepmd.common import (
    j_loader,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

log = logging.getLogger(__name__)


@dataclass
class TrainEntrypointOptions:
    """Common command options for backend train entrypoints."""

    input_file: str
    output: str = "out.json"
    init_model: str | None = None
    restart: str | None = None
    init_frz_model: str | None = None
    finetune: str | None = None
    model_branch: str = ""
    use_pretrain_script: bool = False
    skip_neighbor_stat: bool = False


class AbstractTrainEntrypoint(ABC):
    """Shared pipeline for backend train entrypoints.

    Backend subclasses keep ownership of backend-specific feature handling,
    neighbor-stat updates, distributed setup, data construction, and trainer
    construction.  This pipeline only coordinates the common command flow.
    """

    def run(self, options: TrainEntrypointOptions) -> None:
        """Run the training entrypoint."""
        log.info("Configuration path: %s", options.input_file)
        options = self.prepare_options(options)
        config = self.load_config(options.input_file)
        self.validate_options(config, options)

        config = self.preprocess_config(config, options)
        multi_task = self.is_multi_task(config)
        config = self.update_input(config)
        config = self.normalize_config(config, multi_task=multi_task)

        neighbor_stat = None
        if not options.skip_neighbor_stat:
            config, neighbor_stat = self.update_neighbor_stat(
                config,
                options,
                multi_task=multi_task,
            )

        self.dump_config(config, options.output)
        self.print_summary()

        try:
            self.setup_run(options, config)
            self.run_training(config, options, neighbor_stat)
        finally:
            self.teardown_run(options, config)

    def prepare_options(
        self,
        options: TrainEntrypointOptions,
    ) -> TrainEntrypointOptions:
        """Normalize command options before reading or preprocessing config."""
        return options

    def load_config(self, input_file: str) -> dict[str, Any]:
        """Load the JSON/YAML training config."""
        return j_loader(input_file)

    def validate_options(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> None:
        """Validate backend feature support before mutating the config."""
        return None

    def preprocess_config(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> dict[str, Any]:
        """Apply backend-specific config preprocessing before argcheck."""
        return config

    def is_multi_task(self, config: dict[str, Any]) -> bool:
        """Return whether the config is in multi-task layout."""
        return "model_dict" in config.get("model", {})

    def update_input(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply DeePMD input-version compatibility conversion."""
        return update_deepmd_input(config, warning=True, dump="input_v2_compat.json")

    def normalize_config(
        self,
        config: dict[str, Any],
        *,
        multi_task: bool,
    ) -> dict[str, Any]:
        """Run DeePMD argcheck normalization."""
        return normalize(config, multi_task=multi_task)

    def update_neighbor_stat(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        *,
        multi_task: bool,
    ) -> tuple[dict[str, Any], Any]:
        """Update descriptor selections from neighbor statistics."""
        return config, None

    def dump_config(self, config: dict[str, Any], output: str) -> None:
        """Dump the normalized config used for training."""
        with open(output, "w") as fp:
            json.dump(config, fp, indent=4)

    def print_summary(self) -> None:
        """Print backend summary information."""
        return None

    def setup_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        """Set up backend runtime state before trainer execution."""
        return None

    def teardown_run(
        self,
        options: TrainEntrypointOptions,
        config: dict[str, Any],
    ) -> None:
        """Tear down backend runtime state after trainer execution."""
        return None

    @abstractmethod
    def run_training(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        neighbor_stat: Any,
    ) -> None:
        """Build backend data/trainer objects and run training."""
        raise NotImplementedError
