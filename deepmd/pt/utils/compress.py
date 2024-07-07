import torch

def _layer_0(self, x, w, b):
        return self.filter_activation_fn(torch.matmul(x, w) + b)

def _layer_1(self, x, w, b):
    t = torch.cat([x, x], dim=1)
    return t, self.filter_activation_fn(torch.matmul(x, w) + b) + t

def make_data(self, xx):
    for layer in range(self.layer_size):
        if layer == 0:
            if self.filter_neuron[0] == 1:
                yy = (
                    _layer_0(
                        xx,
                        self.matrix["layer_" + str(layer + 1)],
                        self.bias["layer_" + str(layer + 1)],
                    )
                    + xx
                )
            elif self.filter_neuron[0] == 2:
                tt, yy = _layer_1(
                    xx,
                    self.matrix["layer_" + str(layer + 1)],
                    self.bias["layer_" + str(layer + 1)],
                )
            else:
                yy = _layer_0(
                    xx,
                    self.matrix["layer_" + str(layer + 1)],
                    self.bias["layer_" + str(layer + 1)],
                )
        else:
            tt, zz = _layer_1(
                yy,
                self.matrix["layer_" + str(layer + 1)],
                self.bias["layer_" + str(layer + 1)],
            )
            yy = zz
    vv = zz
    return vv