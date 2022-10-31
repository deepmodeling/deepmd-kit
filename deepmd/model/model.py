from deepmd.env import tf


class Model:
    def init_variables(self,
                       graph : tf.Graph,
                       graph_def : tf.GraphDef,
                       model_type : str = "original_model",
                       suffix : str = "",
    ) -> None:
        """
        Init the embedding net variables with the given frozen model

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        raise RuntimeError("The 'dp train init-frz-model' command do not support this model!")
