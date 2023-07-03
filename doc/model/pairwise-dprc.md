# Pairwise DPRc

In a pairwise DPRc model, the total energy is divided into QM internal energy and the sum of QM/MM energy for each MM residue:

$$ E = E_\text{QM} + \sum_{l} E_\text{QM/MM,l} $$

Thus, the pairwise DPRc model is divided into two sub-[DPRc models](./dprc.md).
`qm_model` is for the QM internal interaction and `qmmm_model` is for the QM/MM interaction.
The configuration for these two models is similar to [the non-pairwise DPRc model](./dprc.md).
It is noted that the [`se_atten` descriptor](./train-se-atten.md) should be used, as it is the only descriptor to support the mixed type.

```json
{
  "model": {
    "type": "pairwise_dprc",
    "type_map": [
      "C",
      "P",
      "O",
      "H",
      "OW",
      "HW"
    ],
    "type_embedding": {
      "neuron": [
        8
      ],
      "precision": "float32"
    },
    "qm_model": {
      "descriptor": {
        "type": "se_atten",
        "stripped_type_embedding": true,
        "sel": 24,
        "rcut_smth": 0.50,
        "rcut": 9.00,
        "attn_layer": 0,
        "neuron": [
          25,
          50,
          100
        ],
        "resnet_dt": false,
        "axis_neuron": 12,
        "precision": "float32",
        "seed": 1
      },
      "fitting_net": {
        "type": "ener",
        "neuron": [
          240,
          240,
          240
        ],
        "resnet_dt": true,
        "precision": "float32",
        "atom_ener": [
          null,
          null,
          null,
          null,
          0.0,
          0.0
        ],
        "seed": 1
      }
    },
    "qmmm_model": {
      "descriptor": {
        "type": "se_atten",
        "stripped_type_embedding": true,
        "sel": 27,
        "rcut_smth": 0.50,
        "rcut": 6.00,
        "attn_layer": 0,
        "neuron": [
          25,
          50,
          100
        ],
        "resnet_dt": false,
        "axis_neuron": 12,
        "set_davg_zero": true,
        "exclude_types": [
          [
            0,
            0
          ],
          [
            0,
            1
          ],
          [
            0,
            2
          ],
          [
            0,
            3
          ],
          [
            1,
            1
          ],
          [
            1,
            2
          ],
          [
            1,
            3
          ],
          [
            2,
            2
          ],
          [
            2,
            3
          ],
          [
            3,
            3
          ],
          [
            4,
            4
          ],
          [
            4,
            5
          ],
          [
            5,
            5
          ]
        ],
        "precision": "float32",
        "seed": 1
      },
      "fitting_net": {
        "type": "ener",
        "neuron": [
          240,
          240,
          240
        ],
        "resnet_dt": true,
        "seed": 1,
        "precision": "float32",
        "atom_ener": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    }
  }
}
```

The pairwise model needs information for MM residues.
The model uses [`aparam`](../data/system.md) with the shape of `nframes x natoms` to get the residue index.
The QM residue should always use `0` as the index.
For example, `0 0 0 1 1 1 2 2 2` means these 9 atoms are grouped into one QM residue and two MM residues.
