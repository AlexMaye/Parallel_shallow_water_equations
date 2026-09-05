OPTIM+=-O2 -march=native

CXX=mpicxx 
CC=mpicxx  
LD=${CXX}
CXXFLAGS+= -Wall -Wextra -pedantic -std=c++17 -I${HDF5_ROOT}/include -fopenmp
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5 

OBJS= parallel_main.o parallel_swe.o swe.o xdmf_writer.o

all: swe

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f swe xdmf *.o *~
