#!/bin/bash

nline_per_set=2000

if test $# -ge 1; then
    nline_per_set=$1
fi

rm -fr set.*
echo nframe is `cat box.raw | wc -l`
echo nline per set is $nline_per_set

split box.raw	 -l $nline_per_set -d -a 3 box.raw
split coord.raw	 -l $nline_per_set -d -a 3 coord.raw
test -f energy.raw && split energy.raw -l $nline_per_set -d -a 3 energy.raw
test -f force.raw  && split force.raw  -l $nline_per_set -d -a 3 force.raw
test -f virial.raw && split virial.raw -l $nline_per_set -d -a 3 virial.raw
test -f atom_ener.raw && split atom_ener.raw -l $nline_per_set -d -a 3 atom_ener.raw
test -f fparam.raw && split fparam.raw -l $nline_per_set -d -a 3 fparam.raw

nset=`ls | grep box.raw[0-9] | wc -l`
nset_1=$(($nset-1))
echo will make $nset sets

for ii in `seq 0 $nset_1`
do
  echo making set $ii ...
  pi=`printf %03d $ii`
  mkdir set.$pi
  mv box.raw$pi		set.$pi/box.raw
  mv coord.raw$pi	set.$pi/coord.raw
  test -f energy.raw$pi && mv energy.raw$pi set.$pi/energy.raw
  test -f force.raw$pi  && mv force.raw$pi  set.$pi/force.raw
  test -f virial.raw$pi && mv virial.raw$pi set.$pi/virial.raw
  test -f atom_ener.raw$pi && mv atom_ener.raw$pi set.$pi/atom_ener.raw
  test -f fparam.raw$pi && mv fparam.raw$pi set.$pi/fparam.raw

  cd set.$pi
  python -c 'import numpy as np; data = np.loadtxt("box.raw"   , ndmin = 2); data = data.astype (np.float32); np.save ("box",    data)'
  python -c 'import numpy as np; data = np.loadtxt("coord.raw" , ndmin = 2); data = data.astype (np.float32); np.save ("coord",  data)'
  python -c \
'import numpy as np; import os.path; 
if os.path.isfile("energy.raw"): 
   data = np.loadtxt("energy.raw"); 
   data = data.astype (np.float32); 
   np.save ("energy", data)
'
  python -c \
'import numpy as np; import os.path; 
if os.path.isfile("force.raw" ): 
   data = np.loadtxt("force.raw", ndmin = 2); 
   data = data.astype (np.float32); 
   np.save ("force",  data)
'
  python -c \
'import numpy as np; import os.path; 
if os.path.isfile("virial.raw"): 
   data = np.loadtxt("virial.raw", ndmin = 2); 
   data = data.astype (np.float32); 
   np.save ("virial", data)
'
  python -c \
'import numpy as np; import os.path; 
if os.path.isfile("atom_ener.raw"): 
   data = np.loadtxt("atom_ener.raw", ndmin = 2); 
   data = data.astype (np.float32); 
   np.save ("atom_ener", data)
'
  python -c \
'import numpy as np; import os.path; 
if os.path.isfile("fparam.raw"): 
   data = np.loadtxt("fparam.raw", ndmin = 2); 
   data = data.astype (np.float32); 
   np.save ("fparam", data)
'
  rm *.raw
  cd ../
done
