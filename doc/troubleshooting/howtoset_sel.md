# How to set sel ?

The setting of "sel" is related to "rcut" and the coordination number of certain atoms. Some empirical settings on some specific systems are listed below.


system | rcut | sel
---|---|---
Li | 9.0 | [700]
Li | 6.0 | [200]
Si | 6.0 | [300]
water | 6.0 | [200,400]

There is no optimal setting for rcut or sel. They depend on your system and the problem you aim to solve. 

Generally, you may need a larger rcut for metallic systems. As for semiconductor systems, choose a rcut with the third-nearest neighbors included is usually good enough. And as for molecular systems, they are more complex and circumstance specific.
