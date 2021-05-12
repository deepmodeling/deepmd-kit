import os, socket

def get_resource ():
    nodename = socket.gethostname()
    nodelist = [nodename]
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpus is not None :
        if gpus != "" :
            gpus = gpus.split(",")
            gpus = [ii for ii in gpus]
        else :
            gpus = []
    return nodename, nodelist, gpus
