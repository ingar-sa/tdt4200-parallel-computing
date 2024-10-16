# Compiler settings
NVCC        := nvcc
C         := g
NVCCFLAGS   := -O2 -G -g
CXXFLAGS    := -O2 -Wall -g

# CUDA settings
CUDA_PATH   ?= /usr/local/cuda
CUDA_INC    := $(CUDA_PATH)/include
CUDA_LIB    := $(CUDA_PATH)/lib64

# Includes and libraries
INCLUDES    := -I$(CUDA_INC) -I.
LIBRARIES   := -L$(CUDA_LIB) -lcudart

# Project files
TARGET      := cuda_program
CU_SOURCES  := kernel.cu
CPP_SOURCES := main.cpp
CU_OBJECTS  := $(CU_SOURCES:.cu=.o)
CPP_OBJECTS := $(CPP_SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(CU_OBJECTS) $(CPP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBRARIES)

# CUDA compilation
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
