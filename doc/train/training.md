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

By default, the verbosity level of the DeePMD-kit is `INFO`, one may see a lot of important information on the code and environment showing on the screen. Among them two pieces of information regarding data systems worth special notice. 
```bash
DEEPMD INFO    ---Summary of DataSystem: training     -----------------------------------------------
DEEPMD INFO    found 3 system(s):
DEEPMD INFO                                        system  natoms  bch_sz   n_bch   prob  pbc
DEEPMD INFO                         ../data_water/data_0/     192       1      80  0.250    T
DEEPMD INFO                         ../data_water/data_1/     192       1     160  0.500    T
DEEPMD INFO                         ../data_water/data_2/     192       1      80  0.250    T
DEEPMD INFO    --------------------------------------------------------------------------------------
DEEPMD INFO    ---Summary of DataSystem: validation   -----------------------------------------------
DEEPMD INFO    found 1 system(s):
DEEPMD INFO                                        system  natoms  bch_sz   n_bch   prob  pbc
DEEPMD INFO                          ../data_water/data_3     192       1      80  1.000    T
DEEPMD INFO    --------------------------------------------------------------------------------------
```
The DeePMD-kit prints detailed informaiton on the training and validation data sets. The data sets are defined by `"training_data"` and `"validation_data"` defined in the `"training"` section of the input script. The training data set is composed by three data systems, while the validation data set is composed by one data system. The number of atoms, batch size, number of batches in the system and the probability of using the system are all shown on the screen. The last column presents if the periodic boundary condition is assumed for the system. 

During the training, the error of the model is tested every `disp_freq` training steps with the batch used to train the model and with `numb_btch` batches from the validating data. The training error and validation error are printed correspondingly in the file `disp_file` (default is `lcurve.out`). The batch size can be set in the input script by the key `batch_size` in the corresponding sections for training and validation data set. An example of the output 
```bash
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn         lr
      0      3.33e+01    3.41e+01      1.03e+01    1.03e+01      8.39e-01    8.72e-01    1.0e-03
    100      2.57e+01    2.56e+01      1.87e+00    1.88e+00      8.03e-01    8.02e-01    1.0e-03
    200      2.45e+01    2.56e+01      2.26e-01    2.21e-01      7.73e-01    8.10e-01    1.0e-03
    300      1.62e+01    1.66e+01      5.01e-02    4.46e-02      5.11e-01    5.26e-01    1.0e-03
    400      1.36e+01    1.32e+01      1.07e-02    2.07e-03      4.29e-01    4.19e-01    1.0e-03
    500      1.07e+01    1.05e+01      2.45e-03    4.11e-03      3.38e-01    3.31e-01    1.0e-03
```
The file contains 8 columns, form right to left, are the training step, the validation loss, training loss, root mean square (RMS) validation error of energy, RMS training error of energy, RMS validation error of force, RMS training error of force and the learning rate. The RMS error (RMSE) of the energy is normalized by number of atoms in the system. One can visualize this file by a simple Python script:

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

Checkpoints will be written to files with prefix `save_ckpt` every `save_freq` training steps. 

## Warning
It is warned that the example water data (in folder `examples/water/data`) is of very limited amount, is provided only for testing purpose, and should not be used to train a productive model.