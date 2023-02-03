# Descriptor `"hybrid"`

This descriptor hybridizes multiple descriptors to form a new descriptor. For example, we have a list of descriptors denoted by $\mathcal D_1$, $\mathcal D_2$, ..., $\mathcal D_N$, the hybrid descriptor this the concatenation of the list, i.e. $\mathcal D = (\mathcal D_1, \mathcal D_2, \cdots, \mathcal D_N)$.

To use the descriptor in DeePMD-kit, one firstly set the {ref}`type <model/descriptor/type>` to {ref}`hybrid <model/descriptor[hybrid]>`, then provide the definitions of the descriptors by the items in the `list`,
```json
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
