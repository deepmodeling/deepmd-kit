# Freeze a model

The trained neural network is extracted from a checkpoint and dumped into a database. This process is called "freezing" a model. The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc). To freeze a model, typically one does
```bash
$ dp freeze -o graph.pb
```
in the folder where the model is trained. The output database is called `graph.pb`.