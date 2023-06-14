# Freeze a model

The trained neural network is extracted from a checkpoint and dumped into a protobuf(.pb) file. This process is called "freezing" a model. The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc). To freeze a model, typically one does
```bash
$ dp freeze -o graph.pb
```
in the folder where the model is trained. The output model is called `graph.pb`.

In [multi-task mode](../train/multi-task-training.md):
- This process will in default output several models, each of which contains the common descriptor and
one of the user-defined fitting nets in {ref}`fitting_net_dict <model/fitting_net_dict>`, let's name it `fitting_key`, together frozen in `graph_{fitting_key}.pb`.
Those frozen models are exactly the same as single-task output with fitting net `fitting_key`.
- If you add `--united-model` option in this situation,
the total multi-task model will be frozen into one unit `graph.pb`, which is mainly for multi-task initialization and can not be used directly for inference.
