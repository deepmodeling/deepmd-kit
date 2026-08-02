# SPDX-License-Identifier: LGPL-3.0-or-later
"""Skeleton shared by every model class a ``dp test`` run can evaluate."""

import logging
import os
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    ClassVar,
)

import numpy as np

from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
)

log = logging.getLogger(__name__)

__all__ = [
    "ChunkContext",
    "ModelTester",
    "save_txt_file",
    "test_chunk_atoms",
]

DEFAULT_TEST_CHUNK_ATOMS = 1_000_000


def test_chunk_atoms() -> int:
    """Return the number of atoms a test evaluates at once.

    Testing bounds the number of atoms sent to the evaluator at once. Lazy
    data sources such as LMDB also decode only this many atoms, while ordinary
    ``DeepmdData`` systems materialize the full test system before iteration.
    The bound is expressed in atoms, as the evaluation batch size is, so that
    it means the same amount of work whatever the size of a frame.
    ``DP_TEST_CHUNK_ATOMS`` overrides it.

    Returns
    -------
    int
        The maximum number of atoms per chunk, at least one.
    """
    return max(1, int(os.environ.get("DP_TEST_CHUNK_ATOMS", DEFAULT_TEST_CHUNK_ATOMS)))


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
) -> None:
    """Save numpy array to test file.

    Parameters
    ----------
    fname : Path
        File to write.
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended instead of overwriting, by default False
    """
    write_header = (
        "" if append and fname.exists() and fname.stat().st_size > 0 else header
    )
    flags = "a" if append else "w"
    with fname.open(flags, encoding="utf-8") as fp:
        np.savetxt(fp, data, header=write_header)


@dataclass(frozen=True)
class ChunkContext:
    """Where one chunk sits within the system being tested.

    Attributes
    ----------
    system : str
        System label recorded in the detail files.
    detail_file : str or None
        File the per-frame details are written to, or ``None`` to write none.
    append_detail : bool
        Whether the details of this chunk extend an existing file rather than
        starting one.
    frame_offset : int
        Index of the first frame of the chunk within its system, which keeps
        per-frame detail files numbered consistently across chunks.
    detail_group : int
        Zero-based test-group index. The first group keeps the historical
        detail filenames; later groups include this index to avoid mixing
        data from different systems or LMDB subgroups.
    """

    system: str
    detail_file: str | None
    append_detail: bool
    frame_offset: int
    detail_group: int = 0

    @property
    def detail_path(self) -> Path | None:
        """The detail file as a path, or ``None`` when details are not kept."""
        return None if self.detail_file is None else Path(self.detail_file)


class ModelTester(ABC):
    """Evaluate one system of one model class, one chunk at a time.

    A system is walked in evaluation chunks. Lazy data sources such as LMDB
    decode only the current chunk; ordinary ``DeepmdData`` systems materialize
    the complete test system first. The errors of the chunks combine into the
    errors of the system exactly, because an MAE and an RMSE are both recovered
    from partial results weighted by the number of elements each was taken
    over; see :func:`~deepmd.utils.weight_avg.merge_weighted_errors`.

    A subclass supplies only what distinguishes its model class: the labels a
    chunk must carry, how a chunk is evaluated, and how the resulting
    quantities are named and reported.

    Parameters
    ----------
    dp : Any
        The evaluator of the model under test.
    atomic : bool
        Whether per-atom quantities are computed.
    """

    #: ``(quantity, log template)`` pairs, in the order they are reported. A
    #: quantity the run did not produce is absent from the errors and is
    #: therefore skipped.
    report: ClassVar[tuple[tuple[str, str], ...]] = ()
    #: Line closing the report, if the model class needs one.
    report_footer: ClassVar[str | None] = None
    #: Quantities reported per system but withheld from the run-level average.
    per_system_only: ClassVar[tuple[str, ...]] = ()

    def __init__(self, dp: Any, *, atomic: bool) -> None:
        self.dp = dp
        self.atomic = atomic

    @abstractmethod
    def add_data_requirements(self, data: DeepmdData) -> None:
        """Declare the labels every chunk of the system must carry.

        Parameters
        ----------
        data : DeepmdData
            The system about to be tested.
        """

    @abstractmethod
    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate one chunk and report the errors over it.

        Parameters
        ----------
        data : DeepmdData
            The system the chunk was drawn from, consulted for its conventions.
        test_data : dict
            One chunk of test data, as yielded by ``data.iter_test``.
        context : ChunkContext
            Where the chunk sits within the system.

        Returns
        -------
        dict[str, tuple[float, float]]
            The ``(error, weight)`` of every quantity the chunk produced.
        """

    def run(
        self,
        data: DeepmdData,
        system: str,
        numb_test: float,
        detail_file: str | None,
        *,
        append_detail: bool = False,
        detail_group: int = 0,
    ) -> dict[str, tuple[float, float]]:
        """Test one system and report its errors.

        Parameters
        ----------
        data : DeepmdData
            The system to test.
        system : str
            System label used in logs and detail files.
        numb_test : float
            Upper bound on the number of frames tested. A non-finite bound
            tests every frame.
        detail_file : str, optional
            File the per-frame details are written to.
        append_detail : bool, optional
            Whether the details of this system extend an existing file.
        detail_group : int, optional
            Zero-based test-group index used to disambiguate detail filenames
            across systems and LMDB subgroups.

        Returns
        -------
        dict[str, tuple[float, float]]
            The ``(error, weight)`` of every quantity that takes part in the
            run-level average.

        Raises
        ------
        RuntimeError
            If the system holds no test frame.
        """
        self.add_data_requirements(data)

        chunk_errors: list[dict[str, tuple[float, float]]] = []
        frames_tested = 0
        append = append_detail
        for chunk in data.iter_test(
            chunk_atoms=test_chunk_atoms(), numb_test=numb_test
        ):
            context = ChunkContext(
                system=system,
                detail_file=detail_file,
                append_detail=append,
                frame_offset=frames_tested,
                detail_group=detail_group,
            )
            chunk_errors.append(self.evaluate_chunk(data, chunk, context))
            frames_tested += chunk["box"].shape[0]
            append = True

        if not chunk_errors:
            raise RuntimeError(f"No test frames found in system {system}.")

        errors = merge_weighted_errors(chunk_errors)
        log.info(f"# number of test data : {frames_tested:d} ")
        self.log_errors(errors)
        return {
            key: value
            for key, value in errors.items()
            if key not in self.per_system_only
        }

    @classmethod
    def log_errors(cls, errors: Mapping[str, tuple[float, float]]) -> None:
        """Report the errors of a system or of a whole run.

        Parameters
        ----------
        errors : Mapping[str, tuple[float, float]]
            The ``(error, weight)`` of each quantity.
        """
        for key, template in cls.report:
            if key in errors:
                log.info(template.format(f"{errors[key][0]:e}"))
        if cls.report_footer is not None:
            log.info(cls.report_footer)


def _detail_output_path(
    context: ChunkContext,
    suffix: str,
    *,
    frame: int | None = None,
) -> Path:
    """Build a detail path unique to its test group and optional frame."""
    detail_path = context.detail_path
    assert detail_path is not None
    group = f".{context.detail_group}" if context.detail_group else ""
    frame_suffix = f".{frame}" if frame is not None else ""
    return detail_path.with_suffix(f"{suffix}{group}{frame_suffix}")


def _write_per_frame_details(
    context: ChunkContext,
    *,
    suffix: str,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Write one detail file per frame of a chunk.

    Parameters
    ----------
    context : ChunkContext
        Where the chunk sits within the system, which numbers the files.
    suffix : str
        Name of the quantity, used in the file suffix and the header.
    reference : np.ndarray
        Reference values with shape ``(nframes, ...)``.
    prediction : np.ndarray
        Predicted values with shape ``(nframes, ...)``.
    """
    assert context.detail_path is not None
    for index in range(reference.shape[0]):
        frame = context.frame_offset + index
        save_txt_file(
            _detail_output_path(context, f".{suffix}.out", frame=frame),
            np.hstack(
                (
                    reference[index].reshape(-1, 1),
                    prediction[index].reshape(-1, 1),
                )
            ),
            header=f"{context.system} - {frame}: data_{suffix} pred_{suffix}",
            append=False,
        )
