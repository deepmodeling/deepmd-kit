# DPA4 freeze-time inference policy

Read this reference before exporting a DPA4 `.pt2`. The generated archive may
embed inference choices, so do not inherit unknown values of
`DP_TRITON_INFER`, `DP_TF32_INFER`, or `DP_AMP_INFER` from the shell.

Choose one explicit policy for the target node. A conservative production
baseline that avoids the slow default while retaining full-precision
accumulation is:

```bash
export DP_TRITON_INFER=1
export DP_TF32_INFER=0
export DP_AMP_INFER=0
dp --pt freeze -c model.ckpt.pt -o frozen_model
```

`DP_TRITON_INFER=2` autotunes for the current hardware and therefore reinforces
the requirement to freeze and run on the same physical node. Levels 1 and 2 keep
FP32 accumulation. Level 3, TF32, and AMP change the numerical policy and require
task-specific accuracy and stability validation before production.

Record all three values, device/runtime identity, input and output hashes, freeze
command, log, and true exit code with the artifact.
