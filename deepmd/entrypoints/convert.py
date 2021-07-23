from deepmd.utils.convert import convert_13_to_20, convert_12_to_20

def convert(
    *,
    FROM: str,
    input_model: str,
    output_model: str,
    **kwargs,
):
    if FROM == '1.2':
        convert_12_to_20(input_model, output_model)
    elif FROM == '1.3':
        convert_13_to_20(input_model, output_model)
    else:
        raise RuntimeError('unsupported model version ' + FROM)
