# Do we need to set rcut < half boxsize?

When seeking the neighbors of atom i under periodic boundary conditions, DeePMD-kit considers all j atoms within cutoff rcut from atom i in all mirror cells.

So, there is no limitation on the setting of rcut.

PS: The reason why some software requires rcut < half box size is that they only consider the nearest mirrors from the center cell. DeePMD-kit is different from them.
