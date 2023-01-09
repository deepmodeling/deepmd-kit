# Multi-task training

Training on multiple data sets (each data set contains several data systems) can be performed in multi-task mode, 
with one common descriptor and multiple specific fitting nets for each data set. 
One can simply switch the following parameters in training input script to perform multi-task mode:
- {ref}`fitting_net <model/fitting_net>` --> {ref}`fitting_net_dict <model/fitting_net_dict>`, 
each key of which can be one individual fitting net.
- {ref}`training_data <training/training_data>`,  {ref}`validation_data <training/validation_data>` 
--> {ref}`data_dict <training/data_dict>`, each key of which can be one individual data set contains 
several data systems for corresponding fitting net, the keys must be consistent with those in 
{ref}`fitting_net_dict <model/fitting_net_dict>`.
- {ref}`loss <loss>` --> {ref}`loss_dict <loss_dict>`, each key of which can be one individual loss setting 
for corresponding fitting net, the keys must be consistent with those in 
{ref}`fitting_net_dict <model/fitting_net_dict>`, if not set, the corresponding fitting net will use the default loss.
- (Optional) {ref}`fitting_weight <training/fitting_weight>`, each key of which can be a non-negative integer or float, 
deciding the chosen probability for corresponding fitting net in training, if not set or invalid, 
the corresponding fitting net will not be used.

The training procedure will automatically choose single-task or multi-task mode, based on the above parameters. 
Note that parameters of single-task mode and multi-task mode can not be mixed.

The supported descriptors for multi-task mode are listed:
- {ref}`se_a (se_e2_a) <model/descriptor[se_e2_a]>`
- {ref}`se_r (se_e2_r) <model/descriptor[se_e2_r]>`
- {ref}`se_at (se_e3) <model/descriptor[se_e3]>`
- {ref}`se_atten <model/descriptor[se_atten]>`
- {ref}`hybrid <model/descriptor[hybrid]>`

The supported fitting nets for multi-task mode are listed:
- {ref}`ener <model/fitting_net[ener]>`
- {ref}`dipole <model/fitting_net[dipole]>`
- {ref}`polar <model/fitting_net[polar]>`

The output of `dp freeze` command in multi-task mode can be seen in [freeze command](../freeze/freeze.md).

## Share layers among energy fitting networks

The multi-task training can be used to train multiple levels of energies (e.g. DFT and CCSD(T)) at the same time.
In this situation, one can set {ref}`model/fitting_net[ener]/layer_name>` to share some of layers among fitting networks.
The architecture of the layers with the same name should be the same.

For example, if one want to share the first and the third layers for two three-hidden-layer fitting networks, the following parameters should be set.
```json
"fitting_net_dict": {
    "ccsd": {
        "neuron": [
            240,
            240,
            240
        ],
        "layer_name": ["l0", null, "l2", null]
    },  
    "wb97m": {
        "neuron": [
            240,
            240,
            240 
        ],
        "layer_name": ["l0", null, "l2", null]
    }   
}
```
