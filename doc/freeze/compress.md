# Compress a model

Once the frozen model is obtained from deepmd-kit, we can get the neural network structure and its parameters (weights, biases, etc.) from the trained model, and compress it in the following way:
```bash
dp compress -i graph.pb -o graph-compress.pb
```
where `-i` gives the original frozen model, `-o` gives the compressed model. Several other command line options can be passed to `dp compress`, which can be checked with
```bash
$ dp compress --help
```
An explanation will be provided
```
usage: dp compress [-h] [-v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-l LOG_PATH]
                   [-m {master,collect,workers}] [-i INPUT] [-o OUTPUT]
                   [-s STEP] [-e EXTRAPOLATE] [-f FREQUENCY]
                   [-c CHECKPOINT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  -v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR,
                        1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -l LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not
                        specified, the logs will only be output to console
                        (default: None)
  -m {master,collect,workers}, --mpi-log {master,collect,workers}
                        Set the manner of logging when running with MPI.
                        'master' logs only on main process, 'collect'
                        broadcasts logs from workers to master and 'workers'
                        means each process will output its own log (default:
                        master)
  -i INPUT, --input INPUT
                        The original frozen model, which will be compressed by
                        the code (default: frozen_model.pb)
  -o OUTPUT, --output OUTPUT
                        The compressed model (default:
                        frozen_model_compressed.pb)
  -s STEP, --step STEP  Model compression uses fifth-order polynomials to
                        interpolate the embedding-net. It introduces two
                        tables with different step size to store the
                        parameters of the polynomials. The first table covers
                        the range of the training data, while the second table
                        is an extrapolation of the training data. The domain
                        of each table is uniformly divided by a given step
                        size. And the step(parameter) denotes the step size of
                        the first table and the second table will use 10 *
                        step as it's step size to save the memory. Usually the
                        value ranges from 0.1 to 0.001. Smaller step means
                        higher accuracy and bigger model size (default: 0.01)
  -e EXTRAPOLATE, --extrapolate EXTRAPOLATE
                        The domain range of the first table is automatically
                        detected by the code: [d_low, d_up]. While the second
                        table ranges from the first table's upper
                        boundary(d_up) to the extrapolate(parameter) * d_up:
                        [d_up, extrapolate * d_up] (default: 5)
  -f FREQUENCY, --frequency FREQUENCY
                        The frequency of tabulation overflow check(Whether the
                        input environment matrix overflow the first or second
                        table range). By default do not check the overflow
                        (default: -1)
  -c CHECKPOINT_FOLDER, --checkpoint-folder CHECKPOINT_FOLDER
                        path to checkpoint folder (default: .)
  -t TRAINING_SCRIPT, --training-script TRAINING_SCRIPT
                        The training script of the input frozen model
                        (default: None)
```
**Parameter explanation**

Model compression, which including tabulating the embedding-net.
The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first sub-table takes the stride(parameter) as it's uniform stride, while the second sub-table takes 10 * stride as it's uniform stride.
The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table's upper boundary(upper) to the extrapolate(parameter) * upper.
Finally, we added a check frequency parameter. It indicates how often the program checks for overflow(if the input environment matrix overflow the first or second table range) during the MD inference.

**Justification of model compression**

Model compression, with little loss of accuracy, can greatly speed up MD inference time. According to different simulation systems and training parameters, the speedup can reach more than 10 times at both CPU and GPU devices. At the same time, model compression can greatly change the memory usage, reducing as much as 20 times under the same hardware conditions.

**Acceptable original model version**

The model compression interface requires the version of deepmd-kit used in original model generation should be `2.0.0-alpha.0` or above. If one has a frozen 1.2 or 1.3 model, one can upgrade it through the `dp convert-from` interface.(eg: ```dp convert-from 1.2/1.3 -i old_frozen_model.pb -o new_frozen_model.pb```) 
