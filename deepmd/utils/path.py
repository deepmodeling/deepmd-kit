import os
from abc import ABC, abstractmethod

import numpy as np

class DPPath(ABC):
    """The path class to data system (DeepmdData)."""
    def __new__(cls, path: str):
        if os.path.isdir(path):
            return DPOSPath
        elif os.path.isfile(path):
            # assume h5 if it is not dir
            # TODO: check if it is a real h5?
            return DPH5Path
        raise OSError("%s not exists" % path)

    @abstractmethod
    def load_numpy_array(path: str) -> np.ndarray:
        """Load NumPy array.
        
        Parameters
        ----------
        path : str
            Path to numpy array relative to the path system
        
        Returns
        -------
        np.ndarray
            The loaded NumPy array
        """

class DPOSPath(DPPath):
    pass


class DPH5Path(DPPath):
    pass