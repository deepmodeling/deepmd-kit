---
name: deepmd-finetune-dpa3
description: Fine-tune a DPA3 model in DeePMD-kit using the PyTorch backend. Use when the user wants to adapt a pre-trained DPA3 model to a new downstream dataset. Supports fine-tuning from a self-trained DPA3 model (.pt checkpoint), from a multi-task pre-trained model, or from a built-in pretrained model downloaded via `dp pretrained download` (e.g., DPA-3.1-3M, DPA-3.2-5M). Covers single-task and multi-task fine-tuning workflows.
compatibility: Requires deepmd-kit with PyTorch backend installed. GPU strongly recommended.
license: LGPL-3.0-or-later
metadata:
  author: iProzd
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Fine-tuning: DPA3

Fine-tune a pre-trained DPA3 model on a downstream dataset. This skill covers three scenarios:

1. Fine-tuning from a self-trained single-task DPA3 model
1. Fine-tuning from a multi-task pre-trained DPA3 model
1. Fine-tuning from a built-in pretrained model (e.g., DPA-3.1-3M, DPA-3.2-5M) downloaded via `dp pretrained download`

## Quick Start

```bash
# Fine-tune from a self-trained model
dp --pt train input.json --finetune pretrained.pt --use-pretrain-script

# Fine-tune from a built-in pretrained model
dp pretrained download DPA-3.2-5M
dp --pt train input.json --finetune /path/to/DPA-3.2-5M.pt --use-pretrain-script --model-branch OMat24
```

## Agent Responsibilities

1. Determine the fine-tuning scenario:
   - Does the user have a self-trained `.pt` model?
   - Does the user want to use a built-in pretrained model (DPA-3.1-3M, DPA-3.2-5M, etc.)?
   - Is the pre-trained model single-task or multi-task?
1. If using a built-in pretrained model, download it first with `dp pretrained download`.
1. Collect the downstream training data paths and element types.
1. Generate the fine-tuning `input.json`.
1. Run fine-tuning and monitor the learning curve.
1. Freeze and test the fine-tuned model.

## Scenario 1: Fine-tune from a Self-trained Single-task Model

When you have trained a DPA3 model yourself and want to adapt it to new data.

### Step 1: Prepare input.json

When using `--use-pretrain-script`, the model architecture is inherited from the pre-trained model. You only need to specify `type_map`, data paths, and training parameters:

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {},
    "fitting_net": {}
  },
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.0001,
    "stop_lr": 3e-06
  },
  "loss": {
    "type": "ener",
    "start_pref_e": 0.2,
    "limit_pref_e": 20,
    "start_pref_f": 100,
    "limit_pref_f": 60,
    "start_pref_v": 0.02,
    "limit_pref_v": 1
  },
  "optimizer": {
    "type": "AdamW",
    "weight_decay": 0.001
  },
  "training": {
    "training_data": {
      "systems": [
        "./downstream_data/train_0",
        "./downstream_data/train_1"
      ],
      "batch_size": 1
    },
    "validation_data": {
      "systems": [
        "./downstream_data/valid_0"
      ],
      "batch_size": 1
    },
    "numb_steps": 200000,
    "gradient_max_norm": 5.0,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "save_freq": 2000
  }
}
```

Fine-tuning tips:

- Use a smaller `start_lr` (e.g., 1e-4) than training from scratch (1e-3).
- Use fewer `numb_steps` since the model is already pre-trained.
- The elements in the downstream data must be a subset of the pre-trained model's `type_map`.

### Step 2: Run Fine-tuning

```bash
dp --pt train input.json --finetune pretrained.pt --use-pretrain-script
```

The `--use-pretrain-script` flag tells DeePMD-kit to inherit the model architecture from the pre-trained model, so the `descriptor` and `fitting_net` sections in `input.json` can be empty.

Without `--use-pretrain-script`, the model section in `input.json` must exactly match the pre-trained model's architecture.

## Scenario 2: Fine-tune from a Multi-task Pre-trained Model

When the pre-trained model was trained with multiple datasets (multi-task training), you can select a specific branch to fine-tune from.

### Check Available Branches

```bash
dp --pt show multitask_pretrained.pt model-branch
```

### Run Fine-tuning from a Specific Branch

```bash
dp --pt train input.json --finetune multitask_pretrained.pt --model-branch CHOSEN_BRANCH --use-pretrain-script
```

If `--model-branch` is not set or set to `RANDOM`, a randomly initialized fitting net will be used.

### Multi-task Fine-tuning (Prevent Forgetting)

To retain knowledge from the pre-trained datasets during fine-tuning, use multi-task fine-tuning. Prepare a multi-task input script:

```json
{
  "model": {
    "shared_dict": {
      "type_map_all": [
        "O",
        "H",
        "C",
        "N"
      ],
      "dpa3_desc": {
        "type": "dpa3",
        "repflow": {}
      }
    },
    "model_dict": {
      "pre_data_1": {
        "type_map": "type_map_all",
        "descriptor": "dpa3_desc",
        "fitting_net": {}
      },
      "pre_data_2": {
        "type_map": "type_map_all",
        "descriptor": "dpa3_desc",
        "fitting_net": {}
      },
      "downstream": {
        "finetune_head": "pre_data_1",
        "type_map": "type_map_all",
        "descriptor": "dpa3_desc",
        "fitting_net": {}
      }
    }
  },
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.0001,
    "stop_lr": 3e-06
  },
  "loss_dict": {
    "pre_data_1": {
      "type": "ener",
      "start_pref_e": 0.2,
      "limit_pref_e": 20,
      "start_pref_f": 100,
      "limit_pref_f": 60
    },
    "pre_data_2": {
      "type": "ener",
      "start_pref_e": 0.2,
      "limit_pref_e": 20,
      "start_pref_f": 100,
      "limit_pref_f": 60
    },
    "downstream": {
      "type": "ener",
      "start_pref_e": 0.2,
      "limit_pref_e": 20,
      "start_pref_f": 100,
      "limit_pref_f": 60
    }
  },
  "training": {
    "model_prob": {
      "pre_data_1": 0.3,
      "pre_data_2": 0.3,
      "downstream": 1.0
    },
    "data_dict": {
      "pre_data_1": {
        "training_data": {
          "systems": [
            "./pre_data_1/train"
          ],
          "batch_size": 1
        }
      },
      "pre_data_2": {
        "training_data": {
          "systems": [
            "./pre_data_2/train"
          ],
          "batch_size": 1
        }
      },
      "downstream": {
        "training_data": {
          "systems": [
            "./downstream/train"
          ],
          "batch_size": 1
        },
        "validation_data": {
          "systems": [
            "./downstream/valid"
          ],
          "batch_size": 1
        }
      }
    },
    "numb_steps": 200000,
    "gradient_max_norm": 5.0,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "save_freq": 2000
  }
}
```

Key points:

- `"finetune_head": "pre_data_1"` specifies which branch the downstream task fine-tunes from.
- `model_prob` controls the sampling probability for each dataset.
- Pre-trained branches continue training in `init-model` mode; the downstream branch fine-tunes from the selected head.

Run:

```bash
dp --pt train multi_input.json --finetune multitask_pretrained.pt
```

Freeze a specific branch:

```bash
dp --pt freeze -o model_downstream.pth --head downstream
```

## Scenario 3: Fine-tune from Built-in Pretrained Models

DeePMD-kit provides built-in pretrained models that can be downloaded directly.

### Step 1: Check Available Models

```bash
dp pretrained download -h
```

Currently available models include:

- `DPA-3.2-5M` — latest large-scale pretrained model
- `DPA-3.1-3M` — 3M parameter DPA3 pretrained model
- `DPA3-Omol-Large` — large organic molecule model

### Step 2: Download the Model

```bash
# Download to default cache directory
dp pretrained download DPA-3.1-3M

# Download to a custom directory
dp pretrained download DPA-3.1-3M --cache-dir ./models
```

The command prints the local path of the downloaded model file on success.

### Step 3: Check Model Branches (if multi-task)

```bash
dp --pt show /path/to/DPA-3.1-3M.pt model-branch
```

### Step 4: Prepare input.json and Run Fine-tuning

The input.json is the same as Scenario 1. Use `--use-pretrain-script` to inherit the model architecture:

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {},
    "fitting_net": {}
  },
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.0001,
    "stop_lr": 3e-06
  },
  "loss": {
    "type": "ener",
    "start_pref_e": 0.2,
    "limit_pref_e": 20,
    "start_pref_f": 100,
    "limit_pref_f": 60,
    "start_pref_v": 0.02,
    "limit_pref_v": 1
  },
  "optimizer": {
    "type": "AdamW",
    "weight_decay": 0.001
  },
  "training": {
    "training_data": {
      "systems": [
        "./my_data/train_0",
        "./my_data/train_1"
      ],
      "batch_size": 1
    },
    "validation_data": {
      "systems": [
        "./my_data/valid_0"
      ],
      "batch_size": 1
    },
    "numb_steps": 200000,
    "gradient_max_norm": 5.0,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "save_freq": 2000
  }
}
```

The meaning of each parameter can be generated through `dp doc-train-input`.
Considering the output RST documentation on the screen is very long, use `grep` to find the documentation of a specific parameter:

```sh
dp doc-train-input | grep -A 7 training/numb_steps
```

Run fine-tuning:

```bash
# Single-task fine-tuning from a specific branch
dp --pt train input.json --finetune /path/to/DPA-3.1-3M.pt --model-branch CHOSEN_BRANCH --use-pretrain-script

# If the pretrained model is single-task, --model-branch is not needed
dp --pt train input.json --finetune /path/to/DPA3-Omol-Large.pt --use-pretrain-script
```

### Step 5: Freeze and Test

```bash
dp --pt freeze -o finetuned_model.pth
dp --pt test -m finetuned_model.pth -s /path/to/test_system -n 30
```

## Fine-tuning Command Reference

| Command                                                                  | Description                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------- |
| `dp pretrained download <MODEL>`                                         | Download a built-in pretrained model              |
| `dp pretrained download <MODEL> --cache-dir <PATH>`                      | Download to a custom directory                    |
| `dp --pt train input.json --finetune <MODEL>.pt`                         | Fine-tune from a pre-trained model                |
| `dp --pt train input.json --finetune <MODEL>.pt --use-pretrain-script`   | Inherit model architecture from pre-trained model |
| `dp --pt train input.json --finetune <MODEL>.pt --model-branch <BRANCH>` | Fine-tune from a specific branch                  |
| `dp --pt train input.json --finetune <MODEL>.pt --model-branch RANDOM`   | Fine-tune with random fitting net                 |
| `dp --pt show <MODEL>.pt model-branch`                                   | List available branches in a multi-task model     |
| `dp --pt freeze -o model.pth`                                            | Freeze the fine-tuned model                       |
| `dp --pt freeze -o model.pth --head <BRANCH>`                            | Freeze a specific branch (multi-task)             |

## Agent Checklist

- [ ] Pre-trained model file exists (downloaded or self-trained)
- [ ] Downstream data elements are a subset of the pre-trained model's `type_map`
- [ ] `--use-pretrain-script` is used if model architecture is unknown
- [ ] Learning rate is reduced compared to training from scratch (e.g., 1e-4 vs 1e-3)
- [ ] For multi-task pretrained models, the correct `--model-branch` is selected
- [ ] Training completes without NaN in `lcurve.out`
- [ ] Fine-tuned model is frozen and tested
- [ ] Test RMSE values are reported to the user

## References

- [Fine-tuning documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/finetuning.html)
- [Pretrained model download](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/pretrained.html)
- [Multi-task training](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/multi-task-training.html)
- [DPA3 descriptor documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa3.html)
- [DeePMD-kit GitHub](https://github.com/deepmodeling/deepmd-kit)
