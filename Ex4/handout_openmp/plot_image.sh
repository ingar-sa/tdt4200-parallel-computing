#! /usr/bin/env bash
SIZE=1024
DATAFILE=$1
IMAGEFILE=`echo $1 | sed s/dat$/png/ | sed s/data/images/`
cat <<END_OF_SCRIPT | gnuplot -
set term png
set output "$IMAGEFILE"
set zrange[-1:1]
splot "$DATAFILE" binary array=${SIZE}x${SIZE} format='%double' with pm3d
END_OF_SCRIPT
