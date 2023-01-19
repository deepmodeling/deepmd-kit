from deepmd.utils.transfer_to_ascend import mix_precision

def transfer_to_ascend(
    *,
    TO: str,
    input_model: str,
    output_model: str,
    **kwargs,
):
    if TO == 'mix_precision':
        mix_precision(input_model, output_model, **kwargs)
    else:
        raise RuntimeError('unsupported transfering type' + FROM)