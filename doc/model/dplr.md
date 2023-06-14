# Deep potential long-range (DPLR)

Notice: **The interfaces of DPLR are not stable and subject to change**

The method of DPLR is described in [this paper][1]. One is recommended to read the paper before using the DPLR.

In the following, we take the DPLR model for example to introduce the training and LAMMPS simulation with the DPLR model. The DPLR model is trained in two steps.

## Train a deep Wannier model for Wannier centroids

We use the deep Wannier model (DW) to represent the relative position of the Wannier centroid (WC) with the atom with which it is associated. One may consult the introduction of the [dipole model](train-fitting-tensor.md) for a detailed introduction. An example input `wc.json` and a small dataset `data` for tutorial purposes can be found in
```bash
$deepmd_source_dir/examples/water/dplr/train/
```
It is noted that **the tutorial dataset is not enough for training a productive model**.
Two settings make the training input script different from an energy training input:
```json
	"fitting_net": {
	    "type":		"dipole",
	    "dipole_type":	[0],
	    "neuron":		[128, 128, 128],
	    "seed":		1
	},
```
The type of fitting is set to {ref}`dipole <model/fitting_net[dipole]>`. The dipole is associated with type 0 atoms (oxygens), by the setting `"dipole_type": [0]`. What we trained is the displacement of the WC from the corresponding oxygen atom. It shares the same training input as the atomic dipole because both are 3-dimensional vectors defined on atoms.
The loss section is provided as follows
```json
    "loss": {
	"type":		"tensor",
	"pref":		0.0,
	"pref_atomic":	1.0
    },
```
so that the atomic dipole is trained as labels. Note that the NumPy compressed file `atomic_dipole.npy` should be provided in each dataset.

The training and freezing can be started from the example directory by
```bash
dp train dw.json && dp freeze -o dw.pb
```

## Train the DPLR model

The training of the DPLR model is very similar to the standard short-range DP models. An example input script can be found in the example directory. The following section is introduced to compute the long-range energy contribution of the DPLR model, and modify the short-range DP model by this part.
```json
        "modifier": {
            "type":             "dipole_charge",
            "model_name":       "dw.pb",
            "model_charge_map": [-8],
            "sys_charge_map":   [6, 1],
            "ewald_h":          1.00,
            "ewald_beta":       0.40
        },
```
The {ref}`model_name <model/modifier[dipole_charge]/model_name>` specifies which DW model is used to predict the position of WCs. {ref}`model_charge_map <model/modifier[dipole_charge]/model_charge_map>` gives the amount of charge assigned to WCs. {ref}`sys_charge_map <model/modifier[dipole_charge]/sys_charge_map>` provides the nuclear charge of oxygen (type 0) and hydrogen (type 1) atoms. {ref}`ewald_beta <model/modifier[dipole_charge]/ewald_beta>` (unit $\text{Å}^{-1}$) gives the spread parameter controls the spread of Gaussian charges, and {ref}`ewald_h <model/modifier[dipole_charge]/ewald_h>`  (unit Å) assigns the grid size of Fourier transformation.
The DPLR model can be trained and frozen by (from the example directory)
```bash
dp train ener.json && dp freeze -o ener.pb
```

## Molecular dynamics simulation with DPLR

In MD simulations, the long-range part of the DPLR is calculated by the LAMMPS `kspace` support. Then the long-range interaction is back-propagated to atoms by DeePMD-kit. This setup is commonly used in classical molecular dynamics simulations as the "virtual site". Unfortunately, LAMMPS does not natively support virtual sites, so we have to hack the LAMMPS code, which makes the input configuration and script a little wired.

An example of an input configuration file and script can be found in
```bash
$deepmd_source_dir/examples/water/dplr/lmp/
```

We use `atom_style full` for DPLR simulations. the coordinates of the WCs are explicitly written in the configuration file. Moreover, a virtual bond is established between the oxygens and the WCs to indicate they are associated together. The configuration file containing 128 H2O molecules is thus written as
```

512 atoms
3 atom types
128 bonds
1 bond types

0 16.421037674 xlo xhi
0 16.421037674 ylo yhi
0 16.421037674 zlo zhi
0 0 0 xy xz yz

Masses

1 16
2 2
3 16

Atoms

       1        1 1  6 8.4960699081e+00 7.5073699951e+00 9.6371297836e+00
       2        2 1  6 4.0597701073e+00 6.8156299591e+00 1.2051420212e+01
...

     385        1 3 -8 8.4960699081e+00 7.5073699951e+00 9.6371297836e+00
     386        2 3 -8 4.0597701073e+00 6.8156299591e+00 1.2051420212e+01
...

Bonds

1 1 1 385
2 1 2 386
...
```
The oxygens and hydrogens are assigned with atom types 1 and 2 (corresponding to training atom types 0 and 1), respectively. The WCs are assigned with atom type 3. We want to simulate heavy water so the mass of hydrogens is set to 2.

An example input script is provided in
```bash
$deepmd_source_dir/examples/water/dplr/lmp/in.lammps
```
Here are some explanations
```lammps
# groups of real and virtual atoms
group           real_atom type 1 2
group           virtual_atom type 3

# bond between real and its corresponding virtual site should be given
# to setup a map between real and virtual atoms. However, no real
# bonded interaction is applied, thus bond_sytle "zero" is used.
pair_style      deepmd ener.pb
pair_coeff      * *
bond_style      zero
bond_coeff      *
special_bonds   lj/coul 1 1 1 angle no
```
Type 1 and 2 (O and H) are `real_atom`s, while type 3 (WCs) are `virtual_atom`s. The model file `ener.pb` stores both the DW and DPLR models, so the position of WCs and the energy can be inferred from it. A virtual bond type is specified by `bond_style zero`. The `special_bonds` command switches off the exclusion of intramolecular interactions.

```lammps
# kspace_style "pppm/dplr" should be used. in addition the
# gewald(1/distance) should be set the same as that used in
# training. Currently only ik differentiation is supported.
kspace_style	pppm/dplr 1e-5
kspace_modify	gewald ${BETA} diff ik mesh ${KMESH} ${KMESH} ${KMESH}
```
The long-range part is calculated by the `kspace` support of LAMMPS. The `kspace_style` `pppm/dplr` is required. The spread parameter set by variable `BETA` should be set the same as that used in training. The `KMESH` should be set dense enough so the long-range calculation is converged.

```lammps
# "fix dplr" set the position of the virtual atom, and spread the
# electrostatic interaction asserting on the virtual atom to the real
# atoms. "type_associate" associates the real atom type its
# corresponding virtual atom type. "bond_type" gives the type of the
# bond between the real and virtual atoms.
fix		0 all dplr model ener.pb type_associate 1 3 bond_type 1
fix_modify	0 virial yes
```
The fix command `dplr` calculates the position of WCs by the DW model and back-propagates the long-range interaction on virtual atoms to real toms.
At this time, the training parameter {ref}`type_map <model/type_map>` will be mapped to LAMMPS atom types.

```lammps
# compute the temperature of real atoms, excluding virtual atom contribution
compute		real_temp real_atom temp
compute		real_press all pressure real_temp
fix		1 real_atom nvt temp ${TEMP} ${TEMP} ${TAU_T}
fix_modify	1 temp real_temp
```
The temperature of the system should be computed from the real atoms. The kinetic contribution in the pressure tensor is also computed from the real atoms. The thermostat is applied to only real atoms. The computed temperature and pressure of real atoms can be accessed by, e.g.
```lammps
fix             thermo_print all print ${THERMO_FREQ} "$(step) $(pe) $(ke) $(etotal) $(enthalpy) $(c_real_temp) $(c_real_press) $(vol) $(c_real_press[1]) $(c_real_press[2]) $(c_real_press[3])" append thermo.out screen no title "# step pe ke etotal enthalpy temp press vol pxx pyy pzz"
```

The LAMMPS simulation can be started from the example directory by
```bash
lmp -i in.lammps
```
If LAMMPS complains that no model file `ener.pb` exists, it can be copied from the training example directory.

The MD simulation lasts for only 20 steps. If one runs a longer simulation, it will blow up, because the model is trained with a very limited dataset for very short training steps, thus is of poor quality.

Another restriction that should be noted is that the energies printed at the zero steps are not correct. This is because at the zero steps the position of the WC has not been updated with the DW model. The energies printed in later steps are correct.



[1]: https://arxiv.org/abs/2112.13327
