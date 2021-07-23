# Train a Deep Potential model using descriptor `"se_e2_r"`

The notation of `se_e2_r` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from the radial information of atomic configurations. The `e2` stands for the embedding with two-atom information. 

A complete training input script of this example can be found in the directory
```bash
$deepmd_source_dir/examples/water/se_e2_r/input.json
```

The training input script is very similar to that of [`se_e2_a`](train-se-e2-a.md#the-training-input-script). The only difference lies in the `descriptor` section
```json=
	"descriptor": {
	    "type":		"se_e2_r",
	    "sel":		[46, 92],
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "neuron":		[5, 10, 20],
	    "resnet_dt":	false,
	    "seed":		1,
	    "_comment": " that's all"
	},
```
The type of the descriptor is set by the key `"type"`.
