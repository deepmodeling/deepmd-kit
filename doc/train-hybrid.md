# Train a Deep Potential model using descriptor `"hybrid"`

This descriptor hybridize multiple descriptors to form a new descriptor. For example we have a list of descriptor denoted by D_1, D_2, ..., D_N, the hybrid descriptor this the concatenation of the list, i.e. D = (D_1, D_2, ..., D_N).

To use the descriptor in DeePMD-kit, one firstly set the `type` to `"hybrid"`, then provide the definitions of the descriptors by the items in the `list`,
```json=
        "descriptor" :{
            "type": "hybrid",
            "list" : [
                {
		    "type" : "se_e2_a",
		    ...		    
                },
                {
		    "type" : "se_e2_r",
		    ...
                }
            ]
        },
```

A complete training input script of this example can be found in the directory
```bash
$deepmd_source_dir/examples/water/hybrid/input.json
```
