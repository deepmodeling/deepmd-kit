# Do we need to set rcut < half boxsize ?

When seeking the neighbors of atom i under periodic boundary condition, deepmd-kit considers all j atoms within cutoff rcut from atom i in all mirror cells.

So, so there is no limitation on the setting of rcut.

PS: The reason why some softwares require rcut < half boxsize is that they only consider the nearest mirrors from the center cell. Deepmd-kit is totally different from them.
