import os, socket

def get_resource ():
    nodename = socket.gethostname()
    nodelist = [nodename]
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpus is not None :
        gpus = gpus.split(",")
        gpus = [int(ii) for ii in gpus]
    return nodename, nodelist, gpus
