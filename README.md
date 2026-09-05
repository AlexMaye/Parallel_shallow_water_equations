# Shallow water equations

This project implements the finite difference methods to solve the shallow water equations in parallel in three different manners:

- Using Cuda, in the folder `cuda`.
- Using MPI (Message passing interface) with blocking communications in the folder `MPI_Blocking`
- Using MPI with asynchronous communications in the folder `MPI_Isend`. In this version, a process is designed to 
    1. Send its halo regions to its neighbours so that they can update their boundary regions,
    2. Update its non-halo regions while waiting to receive the update from other processes,
    3. Receive its neighbours' halo regions, after its previous computations are finished,
    4. Update its boundary regions using the received halo regions.

The grid is divided horizontally between all available processes.
This project was done in Spring 2025 as part of the course MATH-454 Parallel and high-performance computing given by Prof. Pablo Antolin Sanchez.

# How to run
Any recent compiler like gcc or intel is enough. The Makefiles and cluster instructions are provided in the relevant files.

# Results 
There are less than 1% sequential computations left in the code. In fact, both Cuda and asynchronous MPI implementation scale very-well when increasing the number of cores/threads and increasing the size of the problem.
## Cuda
A quadratic speed-up was observed when increasing the number of threads, up to 1,024 threads per block. A large number of blocks was also launched (255) to keep the Streaming Multiprocessor (SM) occupied and not waste resources.
Finally, a larger speed-up was observed when solving larger problems as each resource was used more efficiently.

## Asynchronous MPI
### Strong scaling
For _P_ the number of cores, the computational complexity of the problem scales as _O(nx(ny/P))_, while the communication cost scales as _O(2α log2(P)+2β + 2 · (α + 3βnx))_, where _α_ is the interprocessor latency and _β_ the
inverse of the interprocessor bandwidth.
The communication cost is therefore limited only when an important amount of cores is used and the size of the problem becomes small for each core. 
Indeed, the measured speedup for a 521x523 grid followed closely the values predicted by Amdahl's law for a problem with 0.6% serial part up to 90 cores divided among 3 clusters. The measured speedup was close to 60 for this grid size with 90 cores.
The speedup degraded faster when solving smaller problems when more cores were added,
with the worst measured speedup being 35 when using 90 cores on a 227x229 grid.
In this case, the communication costs became increasingly expensive and each core spent a significant amount of time waiting for a message to arrive before it could continue solving the problem.
### Weak scaling
The efficiency was observed to stabilise when using at least 12 cores.
Blocking communications yielded worse results than non-blocking communications based on the conducted experiments. Moreover, using more cores became more efficient as the problem size increased:
past 12 cores, the measured efficiency varied from 0.73 for 99 cores starting from a single core on a 256x512 grid and 0.47 for 99 cores starting from a single core on a 151x149 grid.

In all cases, using multithreading through OpenMP degraded the results. 
This could be explained by communication latencies that are less well hidden by asynchronous communications in this case.
    