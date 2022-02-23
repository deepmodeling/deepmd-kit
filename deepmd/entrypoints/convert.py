from deepmd.utils.convert import convert_20_to_21, convert_13_to_21, convert_12_to_21 

def convert(
    *,
    FROM: str,
    input_model: str,
    output_model: str,
    **kwargs,
):
    if FROM == '1.2':
        convert_12_to_21(input_model, output_model)
    elif FROM == '1.3':
        convert_13_to_21(input_model, output_model)
    elif FROM == '2.0':
        convert_20_to_21(input_model, output_model)
    else:
        raise RuntimeError('unsupported model version ' + FROM)
