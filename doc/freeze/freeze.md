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

:::{tab-item} Paddle {{ paddle_icon }}

```bash
$ dp --pd freeze -o model
```

in the folder where the model is trained. The output model is called `model.json` and `model.pdiparams`.

In [multi-task mode](../train/multi-task-training-pd.md), you need to choose one available heads (e.g. `CHOSEN_BRANCH`) by `--head`
to specify which model branch you want to freeze:

```bash
$ dp --pd freeze -o model_branch1 --head CHOSEN_BRANCH
```

The output model is called `model_branch1.json`, which is the specifically frozen model with the `CHOSEN_BRANCH` head.

:::

::::
