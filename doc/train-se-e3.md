# Train a Deep Potential model using descriptor `"se_e3"`

The notation of `se_e3` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input (denoted by `e3`).

A complete training input script of this example can be found in the directory
```bash
$deepmd_source_dir/examples/water/se_e3/input.json
```

The training input script is very similar to that of [`se_e2_a`](train-se-e2-a.md#the-training-input-script). The only difference lies in the `descriptor` section
```json=
	"descriptor": {
	    "type":		"se_e3",
	    "sel":		[40, 80],
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "neuron":		[2, 4, 8],
	    "resnet_dt":	false,
	    "seed":		1,
	    "_comment":		" that's all"
	},
```
The type of the descriptor is set by the key `"type"`.
