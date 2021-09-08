# Parallel training

Currently, parallel training is enabled in a sychoronized way with help of [Horovod](https://github.com/horovod/horovod). DeePMD-kit will decide parallel training or not according to MPI context. Thus, there is no difference in your json/yaml input file.

Testing `examples/water/se_e2_a` on a 8-GPU host, linear acceleration can be observed with increasing number of cards.
| Num of GPU cards | Seconds every 100 samples | Samples per second | Speed up |
|  --  | -- | -- | -- |
| 1  | 1.4515 | 68.89 | 1.00 |
| 2  | 1.5962 | 62.65*2 | 1.82 |
| 4  | 1.7635 | 56.71*4 | 3.29 |
| 8  | 1.7267 | 57.91*8 | 6.72 |

To experience this powerful feature, please intall Horovod and [mpi4py](https://github.com/mpi4py/mpi4py) first. For better performance on GPU, please follow tuning steps in [Horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst).
```bash
# With GPU, prefer NCCL as communicator.
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip3 install horovod mpi4py
```

If your work in CPU environment, please prepare runtime as below:
```bash
# By default, MPI is used as communicator.
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod mpi4py
```

Horovod works in the data-parallel mode resulting a larger global batch size. For example, the real batch size is 8 when `batch_size` is set to 2 in the input file and you lauch 4 workers. Thus, `learning_rate` is automatically scaled by the number of workers for better convergence. Technical details of such heuristic rule are discussed at [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).

With dependencies installed, have a quick try!
```bash
# Launch 4 processes on the same host
CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 \
    dp train --mpi-log=workers input.json
```

Need to mention, environment variable `CUDA_VISIBLE_DEVICES` must be set to control parallelism on the occupied host where one process is bound to one GPU card.

What's more, 2 command-line arguments are defined to control the logging behvaior.
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