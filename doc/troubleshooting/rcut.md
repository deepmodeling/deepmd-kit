# Do we need to set rcut < half boxsize ?

When seeking the neighbors of atom i under periodic boundary condition, deepmd-kit considers all j atoms within cutoff R<sub>cut</sub> from atom i in all mirror cells.

So there is no limitation on the setting of R<sub>cut</sub>.

PS: The reason why some softwares require R<sub>cut</sub> < half boxsize is that they only consider the nearest mirrors from the center cell. Deepmd-kit is totally different from them.
