from deepmd.env import tf
from deepmd.utils import Plugin, PluginVariant

class Fitting:
    @property
    def precision(self) -> tf.DType:
        """Precision of fitting network."""
        return self.fitting_precision
