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

# Toolkit-backed preselection queries and sets DeePMD's runtime before Torch.
# Both must refer to the same GPU runtime. CPU-only LibTorch and builds without
# a DeePMD toolkit remain supported.
if(USE_CUDA_TOOLKIT AND TARGET torch_hip)
  message(FATAL_ERROR "CUDA toolkit cannot be combined with HIP LibTorch")
elseif(USE_ROCM_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA)
  message(FATAL_ERROR "ROCm toolkit cannot be combined with CUDA LibTorch")
endif()
