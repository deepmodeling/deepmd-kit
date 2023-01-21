# Run MD with LAMMPS

Running an MD simulation with LAMMPS is simpler. In the LAMMPS input file, one needs to specify the pair style as follows

```lammps
pair_style     deepmd graph.pb type_map O H
pair_coeff     * *
```
where `graph.pb` is the file name of the frozen model. `type_map` maps atom names with LAMMPS atom types (integers from 1 to Ntypes).
