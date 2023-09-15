## Linear model

One can linearly combine existing models with arbitrary coefficients:

```json
"model": {
    "type": "linear_ener",
    "models": [
    {
        "type": "frozen",
        "model_file": "model0.pb"
    },
    {
        "type": "frozen",
        "model_file": "model1.pb"
    }
    ],
    "weights": [0.5, 0.5]
},
```

{ref}`weights <model[linear_ener]/weights>` can be a list of floats, `mean`, or `sum`.

To obtain the model, one needs to execute `dp train` to do a zero-step training with {ref}`numb_steps <training/numb_steps>` set to `0`, and then freeze the model with `dp freeze`.
