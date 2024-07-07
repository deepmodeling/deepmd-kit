def get_extra_embedding_net_suffix(type_one_side: bool):
    """Get the extra embedding net suffix according to the value of type_one_side.

    Parameters
    ----------
    type_one_side
        The value of type_one_side

    Returns
    -------
    str
        The extra embedding net suffix
    """
    if type_one_side:
        extra_suffix = "_one_side_ebd"
    else:
        extra_suffix = "_two_side_ebd"
    return extra_suffix