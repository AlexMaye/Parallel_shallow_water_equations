NVCC       = nvcc
CXX       ?= g++
LD=${CXX}
NVCCFLAGS += -O2
CXXFLAGS+=-O2 -Wall -Wextra -Werror -pedantic -std=c++11 -I${HDF5_ROOT}/include -I${CUDA_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5 $(NVCCFLAGS) -L${CUDA_ROOT}/lib64 -lcudart


OBJS= main.o swe.o xdmf_writer.o Gpuswe.o

all: swe

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

Gpuswe.o: Gpuswe.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f swe xdmf *.o *~
