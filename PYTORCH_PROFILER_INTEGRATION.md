# PyTorch C++ Profiler Integration Test

This test demonstrates the PyTorch profiler integration with the C++ backend.

## Usage

1. Set environment variables:
```bash
export DP_ENABLE_PYTORCH_PROFILER=1
export DP_PYTORCH_PROFILER_OUTPUT_DIR=./profiler_results
```

2. Run your DeepMD-kit C++ application

3. Check for profiler output in the specified directory:
```bash
# For single-rank or non-MPI usage
ls -la ./profiler_results/pytorch_profiler_trace.json

# For MPI usage, each rank gets its own file
ls -la ./profiler_results/pytorch_profiler_trace_rank*.json
```

## Environment Variables

- `DP_ENABLE_PYTORCH_PROFILER`: Set to `1` or `true` to enable profiling
- `DP_PYTORCH_PROFILER_OUTPUT_DIR`: Directory for profiler output (default: `./profiler_output`)

## Implementation Details

The profiler uses PyTorch's modern `torch::profiler` API and automatically:
- Creates the output directory if it doesn't exist
- Profiles all forward pass operations in DeepPotPT and DeepSpinPT
- Saves profiling results to a JSON file when the object is destroyed
- Automatically includes MPI rank in filename when MPI is available and initialized

## Output Files

- **Single-rank or non-MPI usage**: `pytorch_profiler_trace.json`
- **MPI usage**: `pytorch_profiler_trace_rank{rank}.json` (e.g., `pytorch_profiler_trace_rank0.json`, `pytorch_profiler_trace_rank1.json`)

This ensures that each MPI rank saves its profiling data to a separate file, preventing conflicts in multi-rank simulations.

This is intended for development and debugging purposes.