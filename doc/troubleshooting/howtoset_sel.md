# How to set sel ?

sel_a[i] is a list of integers. The length of the list should be the same as the number of atom types in the system. 

sel_a[i] gives the selected number of type-i neighbors. The full relative coordinates of the neighbors are used by the descriptor.

The setting of "sel" is related to "rcut" and the coordination number of certain atoms. Some empirical settings on some specific systems are listed below.

system | rcut | sel
---|---|---
Li | 9.0 | [700]
Li | 6.0 | [200]
Si | 6.0 | [300]
water | 6.0 | [200,400]
