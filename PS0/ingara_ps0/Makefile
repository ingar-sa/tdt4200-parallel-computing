# Compiler selection
# You can override this by running 'make CC=clang'
CC ?= gcc

CFLAGS = -Wall -Wextra -O3
SRCS = main.c bitmap.c
OBJS = $(SRCS:.c=.o)
TARGET = bitmap

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c bitmap.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) after.bmp

.PHONY: all clean
