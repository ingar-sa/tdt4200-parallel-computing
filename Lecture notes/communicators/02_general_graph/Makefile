CC=mpicc
CFLAGS+= -std=c99
TARGETS=tree_communicator 
NEIGHBORFILES=$(shell ls neighbors_*.txt 2>/dev/null)
.PHONY: all clean
all: ${TARGETS}
graph.dot: ${NEIGHBORFILES}
	echo "digraph {" > graph.dot 
	cat ${NEIGHBORFILES} >> graph.dot
	echo "}" >> graph.dot
%.png: %.dot
	dot -Tpng $< >$*.png
clean:
	-rm -f ${TARGETS} *.dot *.png neighbors_*.txt
