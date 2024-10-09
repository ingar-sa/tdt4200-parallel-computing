CC=gcc
CFLAGS+= -std=c99 -pthread
LDLIBS+= -lm
SEQUENTIAL_SRC_FILES=wave_2d_sequential.c
PARALLEL_SRC_FILES=wave_2d_pthread.c
IMAGES=$(shell find data -type f | sed s/\\.dat/.png/g | sed s/data/images/g )
.PHONY: all clean dirs plot movie
all: dirs ${SEQUENTIAL_SRC_FILES} ${PARALLEL_SRC_FILES} 
dirs:
	mkdir -p data images
sequential: ${SEQUENTIAL_SRC_FILES}
	$(CC) $^ $(CFLAGS) -o $@ $(LDLIBS)
parallel: ${PARALLEL_SRC_FILES}
	$(CC) $^ $(CFLAGS) -o $@ $(LDLIBS)
plot: ${IMAGES}
images/%.png: data/%.dat
	./plot_image.sh $<
movie: ${IMAGES}
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4
check: dirs sequential parallel
	mkdir -p data_sequential
	./sequential
	cp -rf ./data/* ./data_sequential
	./parallel 1
	./compare.sh
	rm ./data/*
	./parallel 4
	./compare.sh
	rm ./data/*
	./parallel 13
	./compare.sh
	rm -rf data_sequential
clean:
	-rm -fr ${TARGETS} data images wave.mp4
	-rm sequential
	-rm parallel
