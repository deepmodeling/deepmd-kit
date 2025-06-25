# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

from deepmd.jax.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)


def freeze(
    *,
    checkpoint_folder: str,
    output: str,
    hessian: bool = False,
    **kwargs,
) -> None:
    """Freeze the graph in supplied folder.

    Parameters
    ----------
    checkpoint_folder : str
        location of either the folder with checkpoint or the checkpoint prefix
    output : str
        output file name
    hessian : bool, optional
        whether to freeze the hessian, by default False
    **kwargs
        other arguments
    """
    if (Path(checkpoint_folder) / "checkpoint").is_file():
        checkpoint_meta = Path(checkpoint_folder) / "checkpoint"
        checkpoint_folder = checkpoint_meta.read_text().strip()
    if Path(checkpoint_folder).is_dir():
        data = serialize_from_file(checkpoint_folder)
        deserialize_to_file(output, data, hessian=hessian)
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_folder} does not exist.")
