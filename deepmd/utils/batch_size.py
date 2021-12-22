import logging
from typing import Callable, Tuple

import numpy as np

from deepmd.utils.errors import OutOfMemoryError

class AutoBatchSize:
    """This class allows DeePMD-kit to automatically decide the maximum
    batch size that will not cause an OOM error.
    
    Notes
    -----
    We assume all OOM error will raise :metd:`OutOfMemoryError`.

    Parameters
    ----------
    initial_batch_size : int, default: 1024
        initial batch size (number of total atoms)
    factor : float, default: 2.
        increased factor

    Attributes
    ----------
    current_batch_size : int
        current batch size (number of total atoms)
    maximum_working_batch_size : int
        maximum working batch size 
    minimal_not_working_batch_size : int
        minimal not working batch size
    """
    def __init__(self, initial_batch_size: int = 1024, factor: float = 2.) -> None:
        # See also PyTorchLightning/pytorch-lightning#1638
        # TODO: discuss a proper initial batch size
        self.current_batch_size = initial_batch_size
        self.maximum_working_batch_size = 0
        self.minimal_not_working_batch_size = 2**31
        self.factor = factor

    def execute(self, callable: Callable, start_index: int, natoms: int) -> Tuple[int, tuple]:
        """Excuate a method with given batch size.
        
        Parameters
        ----------
        callable : Callable
            The method should accept the batch size and start_index as parameters,
            and returns executed batch size and data.
        start_index : int
            start index
        natoms : int
            natoms
        
        Returns
        -------
        int
            executed batch size * number of atoms
        tuple
            result from callable, None if failing to execute

        Raises
        ------
        OutOfMemoryError
            OOM when batch size is 1
        """
        try:
            n_batch, result = callable(max(self.current_batch_size // natoms, 1), start_index)
        except OutOfMemoryError as e:
            # TODO: it's very slow to catch OOM error; I don't know what TF is doing here
            # but luckily we only need to catch once
            self.minimal_not_working_batch_size = min(self.minimal_not_working_batch_size, self.current_batch_size)
            if self.maximum_working_batch_size >= self.minimal_not_working_batch_size:
                self.maximum_working_batch_size = int(self.minimal_not_working_batch_size / self.factor)
            if self.minimal_not_working_batch_size <= natoms:
                raise OutOfMemoryError("The callable still throws an out-of-memory (OOM) error even when batch size is 1!") from e
            # adjust the next batch size
            self._adjust_batch_size(1./self.factor)
            return 0, None
        else:
            n_tot = n_batch * natoms
            self.maximum_working_batch_size = max(self.maximum_working_batch_size, n_tot)
            # adjust the next batch size
            if n_tot + natoms > self.current_batch_size and self.current_batch_size * self.factor < self.minimal_not_working_batch_size:
                self._adjust_batch_size(self.factor)
            return n_batch, result

    def _adjust_batch_size(self, factor: float):
        old_batch_size = self.current_batch_size
        self.current_batch_size = int(self.current_batch_size * factor)
        logging.info("Adjust batch size from %d to %d" % (old_batch_size, self.current_batch_size))

    def execute_all(self, callable: Callable, total_size: int, natoms: int, *args, **kwargs) -> Tuple[np.ndarray]:
        """Excuate a method with all given data. 
        
        Parameters
        ----------
        callable : Callable
            The method should accept *args and **kwargs as input and return the similiar array.
        total_size : int
            Total size
        natoms : int
            The number of atoms
        **kwargs
            If 2D np.ndarray, assume the first axis is batch; otherwise do nothing.
        """
        def execute_with_batch_size(batch_size: int, start_index: int) -> Tuple[int, Tuple[np.ndarray]]:
            end_index = start_index + batch_size
            end_index = min(end_index, total_size)
            return (end_index - start_index), callable(
                *[(vv[start_index:end_index] if isinstance(vv, np.ndarray) and vv.ndim > 1 else vv) for vv in args],
                **{kk: (vv[start_index:end_index] if isinstance(vv, np.ndarray) and vv.ndim > 1 else vv) for kk, vv in kwargs.items()},
            )

        index = 0
        results = []
        while index < total_size:
            n_batch, result = self.execute(execute_with_batch_size, index, natoms)
            if not isinstance(result, tuple):
                result = (result,)
            index += n_batch
            if n_batch:
                for rr in result:
                    rr.reshape((n_batch, -1))
                results.append(result)
        
        r = tuple([np.concatenate(r, axis=0) for r in zip(*results)])
        if len(r) == 1:
            # avoid returning tuple if callable doesn't return tuple
            r = r[0]
        return r
