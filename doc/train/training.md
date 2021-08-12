# Training a model

Several examples of training can be found at the `examples` directory:
```bash
$ cd $deepmd_source_dir/examples/water/se_e2_a/
```

After switching to that directory, the training can be invoked by
```bash
$ dp train input.json
```
where `input.json` is the name of the input script.

During the training, checkpoints will be written to files with prefix `save_ckpt` every `save_freq` training steps. The training loss will be saved to `lcurve.out`. One can visualize it by a simple Python script:

```py
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("lcurve.out", names=True)
for name in data.dtype.names[1:-1]:
    plt.plot(data['step'], data[name], label=name)
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.xscale('symlog')
plt.yscale('symlog')
plt.grid()
plt.show()
```
