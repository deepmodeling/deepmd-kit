from deepmd.env import tf

constant_variables = {}

def add_constant_variable(
    key: str,
    var: tf.Tensor
):
    """Store the global constant variables.

    Parameters
    ----------
    key : str
        name of the variable
    var : int
        variables that need to be stored
    """
    constant_variables[key] = var