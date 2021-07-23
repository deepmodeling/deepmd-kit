# How to control the number of nodes used by a job ?

Set the number of CPU nodes used by DP algorithms with:
```bash
mpirun -np $num_nodes dp
```
Set the number of threads used by DP algorithms with:
```bash
export OMP_NUM_THREADS=$num_threads
```

Set the number of CPU nodes used by TF kernels with:
```bash
export TF_INTRA_OP_PARALLELISM_THREADS=$num_nodes
export TF_INTER_OP_PARALLELISM_THREADS=$num_nodes
```
