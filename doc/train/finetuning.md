# Finetune the pretrained model {{ tensorflow_icon }} {{ pytorch_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}
:::

## TensorFlow Implementation {{ tensorflow_icon }}

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

## PyTorch Implementation {{ pytorch_icon }}

In PyTorch version, leveraging the flexibility offered by the framework and the multi-task training capabilities provided by DPA2 [paper](https://arxiv.org/abs/2312.15492),
we have introduced an updated, more adaptable approach to fine-tuning. This methodology encompasses two primary variations:

### 1. Single-task fine-tuning

This fine-tuning method is similar to the fine-tune approach supported by TensorFlow.
It utilizes a standard pre-trained model (`pretrained.pt`, single-task pre-trained) and modifies the energy bias within its fitting net before continuing with training.
The command for this operation is:

```bash
$ dp --pt train input.json --finetune pretrained.pt
```

The model section in input.json can be simplified as follows:

```json
    "model": {
        "type_map":     ["O", "H"],
        "descriptor" :  {},
        "fitting_net" : {}
    }
```

Additionally, within the PyTorch implementation, we also support more general multitask pre-trained models,
which includes multiple datasets for pre-training. These pre-training datasets share a common descriptor while maintaining their individual fitting nets,
as detailed in the DPA2 [paper](https://arxiv.org/abs/2312.15492).
For fine-tuning using this multitask pre-trained model (`multitask_pretrained.pt`),
one can select a specific branch (e.g., `CHOOSEN_BRANCH`) included in `multitask_pretrained.pt` for fine-tuning with the following command:

```bash
$ dp --pt train input.json --finetune multitask_pretrained.pt --model-branch CHOOSEN_BRANCH
```

This command will start fine-tuning based on the pre-trained model's descriptor and the selected branch's fitting net.
If --model-branch is not set, a randomly initialized fitting net will be used.

### 2. Multi-task fine-tuning

In typical scenarios, relying solely on single-task fine-tuning might gradually lead to the forgetting of information from the pre-trained datasets.
In more advanced scenarios, it is desirable for the model to explicitly retain information from the pre-trained data during fine-tuning to prevent forgetting,
which could be more beneficial for fine-tuning.

To achieve this, it is first necessary to clearly identify the datasets from which the pre-trained model originates and to download the corresponding datasets
that need to be retained for subsequent multitask fine-tuning.
Then, prepare a suitable input script for multitask fine-tuning `multi_input.json` as the following steps.

- Suppose the new dataset for fine-tuning is named `DOWNSTREAM_DATA`, and the datasets to be retained from multitask pre-trained model are `PRE_DATA1` and `PRE_DATA2`. One can:

1. Refer to the [`multi-task-training`](./multi-task-training-pt.md) document to prepare a multitask training script for two systems,
   ideally extracting parts (i.e. `model_dict`, `loss_dict`, `data_dict` and `model_prob` parts) corresponding to `PRE_DATA1` and `PRE_DATA2` directly from the training script of the pre-trained model.
2. For `DOWNSTREAM_DATA`, select a desired branch to fine-tune from (e.g., `PRE_DATA1`), copy the configurations of `PRE_DATA1` as the configuration for `DOWNSTREAM_DATA` and insert the corresponding data path into the `data_dict`,
   thereby generating a three-system multitask training script.
3. In the `model_dict` for `DOWNSTREAM_DATA`, specify the branch from which `DOWNSTREAM_DATA` is to fine-tune using:
   `"finetune_head": "PRE_DATA1"`.

The complete `multi_input.json` should appear as follows ("..." means copied from input script of pre-trained model):

```json
  "model": {
    "shared_dict": {
      ...
    },
    "model_dict": {
      "PRE_DATA1": {
        "type_map": ...,
        "descriptor": ...,
        "fitting_net": ...
      },
      "PRE_DATA2": {
        "type_map": ...,
        "descriptor": ...,
        "fitting_net": ...
      },
      "DOWNSTREAM_DATA": {
        "finetune_head": "PRE_DATA1",
        "type_map": ...,
        "descriptor": ...,
        "fitting_net": ...
      },
    }
  },
  "learning_rate": ...,
  "loss_dict": {
      "PRE_DATA1": ...,
      "PRE_DATA2": ...,
      "DOWNSTREAM_DATA": ...
  },
  "training": {
    "model_prob": {
      "PRE_DATA1": 0.5,
      "PRE_DATA2": 0.5,
      "DOWNSTREAM_DATA": 1.0
    },
    "data_dict": {
      "PRE_DATA1": ...,
      "PRE_DATA2": ...,
      "DOWNSTREAM_DATA": {
        "training_data": "training_data_config_for_DOWNSTREAM_DATA",
        "validation_data": "validation_data_config_for_DOWNSTREAM_DATA"
      }
    },
    ...
  }
```

Subsequently, run the command:

```bash
dp --pt train multi_input.json --finetune multitask_pretrained.pt
```

This will initiate multitask fine-tuning, where for branches `PRE_DATA1` and `PRE_DATA2`,
it is akin to continuing training in `init-model` mode, whereas for `DOWNSTREAM_DATA`,
fine-tuning will be based on the fitting net from `PRE_DATA1`.
You can set `model_prob` for each dataset just the same as that in normal multitask training.
