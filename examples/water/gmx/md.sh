#!/bin/bash
export GMX_DEEPMD_INPUT_JSON=input.json
gmx_mpi grompp -f md.mdp -c water.gro -p water.top -o md.tpr -maxwarn 3
gmx_mpi mdrun -deffnm md
gmx_mpi rdf -f md.trr -s md.tpr -o md_rdf.xvg -ref "name OW" -sel "name OW"
