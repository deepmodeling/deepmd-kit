# Query the linked LibTorch build, not DeePMD's selected toolkit: a CPU-only
# LibTorch may be used in a CUDA/ROCm-enabled DeePMD build. CUDA-only fused ops
# still require their narrower capability, while device preselection supports
# both runtimes. ROCm TorchConfig exports torch_hip through Caffe2Targets.
if(TORCH_CUDA_LIBRARIES)
  set(DEEPMD_TORCH_HAS_CUDA TRUE)
else()
  set(DEEPMD_TORCH_HAS_CUDA FALSE)
endif()
if(DEEPMD_TORCH_HAS_CUDA OR TARGET torch_hip)
  set(DEEPMD_TORCH_HAS_GPU TRUE)
else()
  set(DEEPMD_TORCH_HAS_GPU FALSE)
endif()
