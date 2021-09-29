# Parallel training

Currently, parallel training is enabled in a sychoronized way with help of [Horovod](https://github.com/horovod/horovod).
Depend on the number of training processes (according to MPI context) and number of GPU cards avaliable, DeePMD-kit will decide whether to launch the training in parallel (distributed) mode or in serial mode. Therefore, no additional options is specified in your JSON/YAML input file.

## Tuning learning rate

Horovod works in the data-parallel mode, resulting in a larger global batch size. For example, the real batch size is 8 when `batch_size` is set to 2 in the input file and you launch 4 workers. Thus, `learning_rate` is automatically scaled by the number of workers for better convergence. Technical details of such heuristic rule are discussed at [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).

The number of decay steps required to achieve same accuracy can decrease by the number of cards (e.g., 1/2 of steps in the above case), but needs to be scaled manually in the input file.

In some cases, it won't work well when scale learning rate by worker count in a `linear` way. Then you can try `sqrt` or `none` by setting argument `scale_by_worker` like below.
```json
    "learning_rate" :{
        "scale_by_worker": "none",
        "type": "exp"
    }
```

## Scaling test

Testing `examples/water/se_e2_a` on a 8-GPU host, linear acceleration can be observed with increasing number of cards.

| Num of GPU cards | Seconds every 100 samples | Samples per second | Speed up |
|  --  | -- | -- | -- |
| 1  | 1.4515 | 68.89 | 1.00 |
| 2  | 1.5962 | 62.65*2 | 1.82 |
| 4  | 1.7635 | 56.71*4 | 3.29 |
| 8  | 1.7267 | 57.91*8 | 6.72 |

## How to use

Training workers can be launched with `horovodrun`. The following command launches 4 processes on the same host:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 \
    dp train --mpi-log=workers input.json
```

Need to mention, environment variable `CUDA_VISIBLE_DEVICES` must be set to control parallelism on the occupied host where one process is bound to one GPU card.

When using MPI with Horovod, `horovodrun` is a simple wrapper around `mpirun`. In the case where fine-grained control over options passed to `mpirun`, [`mpirun` can be invoked directly](https://horovod.readthedocs.io/en/stable/mpi_include.html), and it will be detected automatically by Horovod, e.g.,
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun -l -launcher=fork -hosts=localhost -np 4 \
    dp train --mpi-log=workers input.json
```
this is sometimes neccessary on HPC environment.

Whether distributed workers are initiated can be observed at the "Summary of the training" section in the log (`world size` > 1, and `distributed`).
```
[0] DEEPMD INFO    ---Summary of the training---------------------------------------
[0] DEEPMD INFO    distributed
[0] DEEPMD INFO    world size:           4
[0] DEEPMD INFO    my rank:              0
[0] DEEPMD INFO    node list:            ['exp-13-57']
[0] DEEPMD INFO    running on:           exp-13-57
[0] DEEPMD INFO    computing device:     gpu:0
[0] DEEPMD INFO    CUDA_VISIBLE_DEVICES: 0,1,2,3
[0] DEEPMD INFO    Count of visible GPU: 4
[0] DEEPMD INFO    num_intra_threads:    0
[0] DEEPMD INFO    num_inter_threads:    0
[0] DEEPMD INFO    -----------------------------------------------------------------
```

## Logging

What's more, 2 command-line arguments are defined to control the logging behvaior when performing parallel training with MPI.
```
optional arguments:
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
```
