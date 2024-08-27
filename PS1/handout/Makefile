CFLAGS+= -std=c99 -O2 -Wall -Wextra
LDLIBS+= -lm
TARGETS=wave_1d
IMAGES=$(shell find data -type f | sed s/\\.dat/.png/g | sed s/data/images/g )
.PHONY: all clean dirs plot movie
all: dirs ${TARGETS}
dirs:
	mkdir -p data images
plot: ${IMAGES}
images/%.png: data/%.dat
	./plot_image.sh $<
movie: ${IMAGES}
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4
clean:
	-rm -fr ${TARGETS} data images wave.mp4
