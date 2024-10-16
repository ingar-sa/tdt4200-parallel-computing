CC=mpicc
CFLAGS+= -std=c99 -g
TARGETS=mondriaan
.PHONY: all clean
all: ${TARGETS}
composition.ppm: composition.data
	echo "P6 630 900 255" | cat - composition.data > composition.ppm
clean:
	-rm -f ${TARGETS} composition.data composition.ppm
