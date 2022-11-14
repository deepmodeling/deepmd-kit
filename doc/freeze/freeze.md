# Freeze a model

The trained neural network is extracted from a checkpoint and dumped into a database. This process is called "freezing" a model. The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc). To freeze a model, typically one does
```bash
$ dp freeze -o graph.pb
```
in the folder where the model is trained. The output database is called `graph.pb`.

In [multi-task mode](../train/multi-task-training.md), this process will output several databases, each of which contains the common descriptor and 
one specific fitting net `fitting_key` and is called `graph_{fitting_key}.pb`. 
Those frozen models are exactly the same as single-task output with fitting net `fitting_key`.