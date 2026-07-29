# Freeze a model

The trained neural network is extracted from a checkpoint and dumped into a model file. This process is called "freezing" a model.
To freeze a model, typically one does

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash
$ dp freeze -o model.pb
```

in the folder where the model is trained. The output model is called `model.pb`.
The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc).
:::

:::{tab-item} TensorFlow 2 {{ tensorflow_icon }}

```bash
$ dp --tf2 freeze -o model.savedmodeltf
```

Run the command in the training folder. By default, it reads the
`model.ckpt.tf2` checkpoint directory and writes the TensorFlow SavedModel to
`model.savedmodeltf`. Use `-c` to select another checkpoint directory or
checkpoint prefix. For a multi-task checkpoint, select a branch with
`--head CHOSEN_BRANCH`.
:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
$ dp --pt freeze -o model.pth
```

in the folder where the model is trained. The output model is called `model.pth`.

In [multi-task mode](../train/multi-task-training), you need to choose one available heads (e.g. `CHOSEN_BRANCH`) by `--head`
to specify which model branch you want to freeze:

```bash
$ dp --pt freeze -o model_branch1.pth --head CHOSEN_BRANCH
```

The output model is called `model_branch1.pth`, which is the specifically frozen model with the `CHOSEN_BRANCH` head.
:::

:::{tab-item} PyTorch-Exportable {{ pytorch_icon }}

```bash
$ dp --pt-expt freeze -c model.ckpt.pt -o model
```

The backend writes `.pte` for the dense neighbor-list lower form and `.pt2` for
the graph lower form. A suffixless output lets DeePMD-kit select the matching
extension. Use `--lower-kind nlist` or `--lower-kind graph` to request a form;
graph-capable DPA models may select the graph form automatically. In multi-task
mode, select a model branch with `--head CHOSEN_BRANCH`.
:::

:::{tab-item} Paddle {{ paddle_icon }}

```bash
$ dp --pd freeze -o model
```

in the folder where the model is trained. The output model is called `model.json` and `model.pdiparams`.

In [multi-task mode](../train/multi-task-training.md), you need to choose one available heads (e.g. `CHOSEN_BRANCH`) by `--head`
to specify which model branch you want to freeze:

```bash
$ dp --pd freeze -o model_branch1 --head CHOSEN_BRANCH
```

The output model is called `model_branch1.json`, which is the specifically frozen model with the `CHOSEN_BRANCH` head.
:::

::::
