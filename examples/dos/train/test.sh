#!/bin/bash

$DP freeze -o frozen_model.pb 


for k in size-80 heat-5-fermi   
do            
        $DP test -m frozen_model.pb -s ../data/111/$k -d comp -a -n 500

        mkdir output-$k
        mv comp* ./output-$k
done
