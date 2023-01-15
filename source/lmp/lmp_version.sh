#!/bin/sh
set -e
# Read LAMMPS version from version.h
version_line=$(grep LAMMPS_VERSION ../version.h)
# extract version
tmp=${version_line#*\"}   # remove prefix ending in "
version=${tmp%\"*}   # remove suffix starting with "
# string to int
date --date="$(printf $version)" +"%Y%m%d"
