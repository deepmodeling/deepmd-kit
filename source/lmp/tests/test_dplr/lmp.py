from lammps import PyLammps
lammps = PyLammps()

lammps.units("si")
lammps.boundary("p p p")
lammps.atom_style("full")
lammps.neighbor("2.0e-11 bin")
lammps.neigh_modify("every 1 delay 0 check no exclude type 1 3")
lammps.read_data("data.si")
lammps.timestep("5e-16")
lammps.fix("1 all nve")
lammps.pair_style("deepmd lrmodel.pb")
lammps.pair_coeff("* *")
# 
# #bond_style zero
# #bond_coeff *
# #special_bonds lj/coul 1 1 1 angle no
# #kspace_style pppm/dplr 1e-5
# #kspace_modify gewald 4e9 diff ik mesh 10 10 10
# #fix 0 all dplr model lrmodel.pb type_associate 1 3 bond_type 1
# 
lammps.run(1)
