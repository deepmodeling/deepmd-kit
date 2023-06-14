# Run MD with LAMMPS

Running an MD simulation with LAMMPS is simpler. In the LAMMPS input file, one needs to specify the pair style as follows

```lammps
pair_style     deepmd graph.pb
pair_coeff     * * O H
```
where `graph.pb` is the file name of the frozen model. `pair_coeff` maps atom names (`O H`) with LAMMPS atom types (integers from 1 to Ntypes, i.e. `1 2`).
