from .convert import convert_pb_to_pbtxt

def detect_model_version(input_model: str):
  convert_pb_to_pbtxt(input_model, "frozen_model.pbtxt")
  version = "undetected"
  with open("frozen_model.pbtxt", "r") as fp:
    file_content = fp.read()
  if file_content.find("DescrptNorot") > -1:
    version = "<= 0.12"
  elif file_content.find("fitting_attr/dfparam") > -1 and file_content.find("fitting_attr/daparam") == -1:
    version = "1.0"
  elif file_content.find("model_attr/model_version") == -1:
    version = "1.1 or 1.2 or 1.3"
  elif file_content.find("string_val: \"1.0\"") > -1:
    version = "2.0"
  elif file_content.find("string_val: \"1.1\"") > -1:
    version = ">= 2.1"
  print("The version of model %s is %s."%(input_model, version))
