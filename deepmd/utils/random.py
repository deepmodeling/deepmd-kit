import numpy as np


_RANDOM_GENERATOR = np.random.RandomState()


def choice(a: np.ndarray, p: np.ndarray = None):
    """Generates a random sample from a given 1-D array.

    Parameters
    ----------
    a : np.ndarray
        A random sample is generated from its elements.
    p : np.ndarray
        The probabilities associated with each entry in a.

    Returns
    -------
    np.ndarray
        arrays with results and their shapes
    """
    return _RANDOM_GENERATOR.choice(a, p=p)


def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    size
        Output shape.

    Returns
    -------
    np.ndarray
        Arrays with results and their shapes.
    """
    return _RANDOM_GENERATOR.random_sample(size)


def seed(val: int = None):
    """Seed the generator.

    Parameters
    ----------
    val : int
        Seed.
    """
    _RANDOM_GENERATOR.seed(val)


def shuffle(x: np.ndarray):
    """Modify a sequence in-place by shuffling its contents.

    Parameters
    ----------
    x : np.ndarray
        The array or list to be shuffled.
    """
    _RANDOM_GENERATOR.shuffle(x)


__all__ = ['choice', 'random', 'seed', 'shuffle']
