#! /usr/bin/env bash
SIZE=1024
DATAFILE=$1
IMAGEFILE=`echo $1 | sed s/dat$/png/ | sed s/data/images/`
cat <<END_OF_SCRIPT | gnuplot -
set term png
set output "$IMAGEFILE"
set yrange[-1:1]
plot "$DATAFILE" binary array=${SIZE} format='%double' with lines
END_OF_SCRIPT
