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
- {ref}`se_a(se_e2_a) <model/descriptor[se_a]>`
- {ref}`se_r(se_e2_r) <model/descriptor[se_r]>`
- {ref}`se_t(se_e2_t) <model/descriptor[se_t]>`
- {ref}`se_atten <model/descriptor[se_atten]>`
- {ref}`hybrid <model/descriptor[hybrid]>`

The supported fitting nets for multi-task mode are listed:
- {ref}`ener <model/fitting_net[se_a]>`
- {ref}`dipole <model/fitting_net[dipole]>`
- {ref}`polar <model/fitting_net[polar]>`

The output of `dp freeze` command in multi-task mode can be seen in [freeze command](../freeze/freeze.md).
