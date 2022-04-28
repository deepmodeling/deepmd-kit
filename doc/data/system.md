# System

DeePMD-kit takes a **system** as data structure. A snapshot of a system is called a **frame**. A system may contain multiple frames with the same atom types and numbers, i.e. the same formula (like `H2O`). To contains data with different formula, one need to divide data into multiple systems.

A system should contain system properties, input frame properties, and labeled frame properties. The system property contains the following property:

ID       | Property                | Required/Optional    | Shape                    | Description
-------- | ----------------------  | -------------------- | -----------------------  | -----------
type     | Atom type indexes       | Required             | Natoms                   | Integers that start with 0
type_map | Atom type names         | Optional             | Ntypes                   | Atom names that map to atom type, which is unnecessart to be contained in the periodic table
nopbc    | Non-periodic system     | Optional             | 1                        | If True, this system is non-periodic; otherwise it's periodic

The input frame properties contains the following property, the first axis of which is the number of frames:

ID       | Property                | Unit | Required/Optional    | Shape                    | Description
-------- | ----------------------  | ---- | -------------------- | -----------------------  | -----------
coord    | Atomic coordinates      | Å    | Required             | Nframes \* Natoms \* 3   | 
box      | Boxes                   | Å    | Required if periodic | Nframes \* 3 \* 3        | in the order `XX XY XZ YX YY YZ ZX ZY ZZ`
fparam   | Extra frame parameters  | Any  | Optional             | Nframes \* Any           |
aparam   | Extra atomic parameters | Any  | Optional             | Nframes \* aparam \* Any |

The labeled frame properties is listed as follows, all of which will be used for training if and only if the loss function contains such property:

ID                     | Property                 | Unit   | Shape                    | Description
---------------------- | -----------------------  | ----   | -----------------------  | -----------
energy                 | Frame energies           | eV     | Nframes                  | 
force                  | Atomic forces            | eV/Å   | Nframes \* Natoms \* 3   | 
virial                 | Frame virial             | eV     | Nframes \* 3             | in the order `XX XY XZ YX YY YZ ZX ZY ZZ`
atom_ener              | Atomic energies          | eV     | Nframes \* Natoms        |
atom_pref              | Weights of atomic forces | 1      | Nframes \* Natoms        |
dipole                 | Frame dipole             | Any    | Nframes \* 3             |
atomic_dipole          | Atomic dipole            | Any    | Nframes \* Natoms \* 3   |
polarizability         | Frame polarizability     | Any    | Nframes \* 9             | in the order `XX XY XZ YX YY YZ ZX ZY ZZ`
atomic_polarizability  | Atomic polarizability    | Any    | Nframes \* Natoms \* 9   | in the order `XX XY XZ YX YY YZ ZX ZY ZZ`

In general, we always use the following convention of units:

Property | Unit 
---------| ----
Time     | ps   
Length   | Å    
Energy   | eV   
Force    | eV/Å 
Virial   | eV   
Pressure | Bar  
