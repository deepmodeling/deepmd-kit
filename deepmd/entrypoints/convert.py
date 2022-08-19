from deepmd.utils.convert import convert_012_to_21, convert_10_to_21, convert_20_to_21, convert_13_to_21, convert_12_to_21, convert_org_to_ascend

def convert(
    *,
    FROM: str,
    input_model: str,
    output_model: str,
    **kwargs,
):
    if FROM == '0.12':
        convert_012_to_21(input_model, output_model)
    elif FROM == '1.0':
        convert_10_to_21(input_model, output_model)
    elif FROM in ['1.1', '1.2']:
        # no difference between 1.1 and 1.2
        convert_12_to_21(input_model, output_model)
    elif FROM == '1.3':
        convert_13_to_21(input_model, output_model)
    elif FROM == '2.0':
        convert_20_to_21(input_model, output_model)
    elif FROM == 'convert_org_to_ascend':
        convert_org_to_ascend(input_model, output_model, **kwargs)
    else:
        raise RuntimeError('unsupported model version ' + FROM)
