CFLAGS+= -std=c99 -O2 -fopenmp
LDLIBS+= -lm
TARGETS=advection_openmp
PNGFILES=$(shell find data/*.dat | sed s/data/images/g | sed s/\.dat/.png/g)
.PHONY: all dirs plots clean
all: dirs ${TARGETS}
dirs:
	mkdir -p data images
plots: ${PNGFILES}
images/%.png: data/%.dat
	./plot_one_result.sh $<
clean:
	-rm -f ${TARGETS} data/*.dat images/*.png
