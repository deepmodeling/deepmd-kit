#apt install libc-dev
cd /home
rm -rf lammps-stable_29Oct2020/
tar -xzvf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src/
cp -r /home/paddle-deepmd/source/build/USER-DEEPMD .
make yes-kspace yes-user-deepmd
make serial -j 20
