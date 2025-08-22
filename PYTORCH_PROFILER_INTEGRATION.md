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
ls -la ./profiler_results/pytorch_profiler_trace.json
```

## Environment Variables

- `DP_ENABLE_PYTORCH_PROFILER`: Set to `1` or `true` to enable profiling
- `DP_PYTORCH_PROFILER_OUTPUT_DIR`: Directory for profiler output (default: `./profiler_output`)

## Implementation Details

The profiler uses PyTorch's `torch::autograd::profiler::RecordProfile` and automatically:
- Creates the output directory if it doesn't exist
- Profiles all forward pass operations in DeepPotPT and DeepSpinPT
- Saves profiling results to a JSON file when the object is destroyed

This is intended for development and debugging purposes.