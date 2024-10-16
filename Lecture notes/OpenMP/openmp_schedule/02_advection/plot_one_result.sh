PNGFILE=`echo $1 | sed s/data/images/ | sed s/\.dat/.png/`
cat <<EOSCRIPT | gnuplot -
set term png
set output "${PNGFILE}"
set zrange [0:1.5]
splot "$1" binary array=256x256 format='%double' with pm3d
EOSCRIPT
