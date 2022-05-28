from deepmd.env import tf
from deepmd.utils import Plugin, PluginVariant

class Fitting:
    @property
    def precision(self) -> tf.DType:
        """Precision of fitting network."""
        return self.fitting_precision

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the fitting net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        
        Notes
        -----
        This method is called by others when the fitting supported initialization from the given variables.
        """
        raise NotImplementedError(
            "Fitting %s doesn't support initialization from the given variables!" % type(self).__name__)
