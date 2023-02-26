# Finetune the pretrained model

Pretraining-and-finetuning is a widely used approach in other fields such as Computer Vision (CV) or Natural Language Processing (NLP)
to vastly reduce the training cost, while it's not trivial in potential models.
Compositions and configurations of data samples or even computational parameters in upstream software (such as VASP)
may be different between the pretrained and target datasets, leading to energy shifts or other diversities of training data.

Recently the emerging of methods such as [DPA-1](https://arxiv.org/abs/2208.08236) has brought us to a new stage where we can
perform similar pretraining-finetuning approaches.
DPA-1 can hopefully learn the common knowledge in the pretrained dataset (especially the `force` information)
and thus reduce the computational cost in downstream training tasks.
If you have a pretrained model `pretrained.pb`
(here we support models using [`se_atten`](../model/train-se-atten.md) descriptor and [`ener`](../model/train-energy.md) fitting net)
on a large dataset (for example, [OC2M](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md) in
DPA-1 [paper](https://arxiv.org/abs/2208.08236)), a finetuning strategy can be performed by simply running:

```bash
$ dp train input.json --finetune pretrained.pb
```

The command above will change the energy bias in the last layer of the fitting net in `pretrained.pb`,
according to the training dataset in input.json.

:::{warning}
Note that the elements in the training dataset must be contained in the pretrained dataset.
:::

The finetune procedure will inherit the model structures in `pretrained.pb`,
and thus it will ignore the model parameters in `input.json`,
such as {ref}`descriptor <model/descriptor>`, {ref}`fitting_net <model/fitting_net>`,
{ref}`type_embedding <model/type_embedding>` and {ref}`type_map <model/type_map>`.
However, you can still set the `trainable` parameters in each part of `input.json` to control the training procedure.

To obtain a more simplified script, for example, you can change the {ref}`model <model>` part in `input.json` to perform finetuning:
```json
    "model": {
        "type_map":     ["O", "H"],
        "type_embedding": {"trainable":  true},
        "descriptor" :  {},
        "fitting_net" : {}
    }
```
