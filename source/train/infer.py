
from deepmd.DeepPot import DeepPot
import numpy as np

def init_model(model_path):
    # print("in init model")
    # print(model_path)
    return DeepPot(model_path)

def get_scalar(model,name):
    # print("in get scalar !!!")
    # print(name)
    # TODO
    name = "load/"+name+":0"
    ret = model.sess.run(model.graph.get_tensor_by_name(name))
    return ret

def infer(model,t_coord,t_type,t_box,t_mesh,t_natoms):
    # print("in infer")

    nframes = t_coord.shape[0]

    energy = []
    force = []
    virial = []
    feed_dict = {}

    feed_dict[model.t_mesh ] = t_mesh
    feed_dict[model.t_natoms] = t_natoms

    t_out = [model.t_energy, 
            model.t_force, 
            model.t_virial]

    for ii in range(nframes) :
        feed_dict[model.t_coord] = np.reshape(t_coord[ii:ii+1, :], [-1])
        feed_dict[model.t_type  ] = np.reshape(t_type[ii:ii+1, :], [-1])
        feed_dict[model.t_box  ] = np.reshape(t_box [ii:ii+1, :], [-1])
        v_out = model.sess.run (t_out, feed_dict = feed_dict)
        energy.append(v_out[0])
        force .append(v_out[1])
        virial.append(v_out[2])

    # reverse map of the outputs
    # force  = self.reverse_map(np.reshape(force, [nframes,-1,3]), imap)

    energy = np.reshape(energy, [nframes, 1])
    force  = np.reshape(force, [nframes,-1,3])
    virial = np.reshape(virial, [nframes, 9])

    return energy, force, virial
