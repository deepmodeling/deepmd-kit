# Multi-task training {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

<!-- we plan to drop TensorFlow backend multi-task training. Replace with the PyTorch one -->

## Theory

The multi-task training process can simultaneously handle different datasets with properties that cannot be fitted in one network (e.g. properties from DFT calculations under different exchange-correlation functionals or different basis sets).
These datasets are denoted by $\boldsymbol x^{(1)}, \dots, \boldsymbol x^{(n_t)}$.
For each dataset, a training task is defined as

```math
    \min_{\boldsymbol \theta}   L^{(t)} (\boldsymbol x^{(t)}; \boldsymbol  \theta^{(t)}, \tau), \quad t=1, \dots, n_t.
```

In the Pytorch implementation, during the multi-task training process, all tasks can share any portion of the model parameters.
A typical scenario is that each task shares the same descriptor with trainable parameters $\boldsymbol{\theta}_ {d}$, while each has its own fitting network with trainable parameters $\boldsymbol{\theta}_ f^{(t)}$, thus
$\boldsymbol{\theta}^{(t)} = \{ \boldsymbol{\theta}_ {d} , \boldsymbol{\theta}_ {f}^{(t)} \}$.
At each training step, a task will be randomly selected from ${1, \dots, n_t}$ according to the user-specified probability,
and the Adam optimizer is executed to minimize $L^{(t)}$ for one step to update the parameter $\boldsymbol \theta^{(t)}$.
In the case of multi-GPU parallel training, different GPUs will independently select their tasks.
In the DPA-2 model, this multi-task training framework is adopted.[^1]

[^1] Duo Zhang, Xinzijian Liu, Xiangyu Zhang, Chengqian Zhang, Chun Cai, Hangrui Bi, Yiming Du, Xuejian Qin, Jiameng Huang, Bowen Li, Yifan Shan, Jinzhe Zeng, Yuzhi Zhang, Siyuan Liu, Yifan Li, Junhan Chang, Xinyan Wang, Shuo Zhou, Jianchuan Liu, Xiaoshan Luo, Zhenyu Wang, Wanrun Jiang, Jing Wu, Yudi Yang, Jiyuan Yang, Manyi Yang, Fu-Qiang Gong, Linshuang Zhang, Mengchao Shi, Fu-Zhi Dai, Darrin M. York, Shi Liu, Tong Zhu, Zhicheng Zhong, Jian Lv, Jun Cheng, Weile Jia, Mohan Chen, Guolin Ke, Weinan E, Linfeng Zhang, Han Wang，[arXiv preprint arXiv:2312.15492 (2023)](https://arxiv.org/abs/2312.15492) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

Compared with the previous TensorFlow implementation, the new support in PyTorch is more flexible and efficient.
In particular, it makes multi-GPU parallel training and even tasks beyond DFT possible,
enabling larger-scale and more general multi-task training to obtain more general pre-trained models.

## Perform the multi-task training using PyTorch

Training on multiple data sets (each data set contains several data systems) can be performed in multi-task mode,
typically with one common descriptor and multiple specific fitting nets for each data set.
To proceed, one need to change the representation of the model definition in the input script.
The core idea is to replace the previous single model definition {ref}`model <model>` with multiple model definitions {ref}`model/model_dict/model_key <model/model_dict/model_key>`,
define the shared parameters of the model part {ref}`shared_dict <model/shared_dict>`, and then expand other parts for multi-model settings.
Specifically, there are several parts that need to be modified:

- {ref}`model/shared_dict <model/shared_dict>`: The parameter definition of the shared part, including various descriptors,
  type maps (or even fitting nets can be shared). Each module can be defined with a user-defined `part_key`, such as `my_descriptor`.
  The content needs to align with the corresponding definition in the single-task training model component, such as the definition of the descriptor.

- {ref}`model/model_dict <model/model_dict>`: The core definition of the model part and the explanation of sharing rules,
  starting with user-defined model name keys `model_key`, such as `my_model_1`.
  Each model part needs to align with the components of the single-task training {ref}`model <model>`, but with the following sharing rules:
- - If you want to share the current model component with other tasks, which should be part of the {ref}`model/shared_dict <model/shared_dict>`,
    you can directly fill in the corresponding `part_key`, such as
    `"descriptor": "my_descriptor", `
    to replace the previous detailed parameters. Here, you can also specify the shared_level, such as
    `"descriptor": "my_descriptor:shared_level", `
    and use the user-defined integer `shared_level` in the code to share the corresponding module to varying degrees
    (default is to share all parameters, i.e., `shared_level`=0).
    The parts that are exclusive to each model can be written following the previous definition.

- {ref}`loss_dict <loss_dict>`: The loss settings corresponding to each task model, specified by the `model_key`.
  Each {ref}`loss_dict/model_key <loss_dict/model_key>` contains the corresponding loss settings,
  which are the same as the definition in single-task training {ref}`<loss>`.

- {ref}`training/data_dict <training/data_dict>`: The data settings corresponding to each task model, specified by the `model_key`.
  Each `training/data_dict/model_key` contains the corresponding `training_data` and `validation_data` settings,
  which are the same as the definition in single-task training {ref}`training_data <training/training_data>` and {ref}`validation_data <training/validation_data>`.

- (Optional) {ref}`training/model_prob <training/model_prob>`: The sampling weight settings corresponding to each `model_key`, i.e., the probability weight in the training step.
  You can specify any positive real number weight for each task. The higher the weight, the higher the probability of being sampled in each training.
  This setting is optional, and if not set, tasks will be sampled with equal weights.

An example input for multi-task training two models in water system is shown as following:

```{literalinclude} ../../examples/water_multi_task/pytorch_example/input_torch.json
:language: json
:linenos:
```

## Finetune from the pretrained multi-task model

To finetune based on the checkpoint `model.pt` after the multi-task pre-training is completed,
users only need to prepare the normal input for single-task training `input_single.json`,
and then select one of the trained model's task names `model_key`.
Run the following command:

```bash
$ dp --pt train input_single.json --finetune model.pt --model-branch model_key
```
