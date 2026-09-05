#include "parallel_swe.hh"
#include "swe.hh"

#include <string>
#include <cstddef>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

#ifdef _OPENMP
 #include <omp.h>
#endif
#include <cstdio>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

template<typename T>
double mean(std::vector<T> a){
  double mmean = 0;
  const std::size_t sizea = a.size();
  for(const auto i:a){
    mmean += (1.*i);
  }
  return mmean/(1.*sizea);
}

template<typename T>
void check_size(std::vector<T>& vec, const int size){
    const std::size_t my_size = size;
    if (vec.size()!=my_size)
        vec.resize(my_size);
}

/// @brief modifies sendcounts and displacements vectors for Scatterv
/// @param ny: number of rows of grid
/// @param nx: nmber of columns of grid
/// @param sendcounts: how many elements each process gets
/// @param displacements: where to start sending
/// @param size: number of cores
void get_sendcounts_displacements(const std::size_t ny, const std::size_t nx, std::vector<int>& sendcounts, std::vector<int>& displacements, const int size){
    check_size(sendcounts, size);
    check_size(displacements, size);
    const int div = ny / size;
    const int remainder = ny % size;
    displacements[0] = 0;
    for(int i = 0; i<size; ++i){
        sendcounts[i] = nx * ((i<remainder) ? (div+1) : div);
        if(i>0)
            displacements[i] = displacements[i-1] + sendcounts[i-1];
    }
}


int main(int argc, char *argv[]){
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  

  constexpr std::size_t measurements = 2;

  constexpr std::size_t vector_size = (measurements>1)? measurements-1 : 1;
  std::vector<double> times(vector_size);
  for(size_t j = 0; j<measurements; ++j){    


    // Option 1 - Solving simple problem: water drops in a box
    constexpr int test_case_id = 1;  // Water drops in a box (1)
    constexpr double Tend = 1.0;     // Simulation time in hours
    const std::size_t nx = std::stoul(argv[1]); // Number of cells per direction.
    const std::size_t ny = std::stoul(argv[2]); // Number of cells per direction.

  //constexpr std::size_t measurements_if_one_run = 10;
    constexpr std::size_t output_n = 0;//(measurements<=1)? measurements_if_one_run : 0;
    const std::string output_fname = "water_drops";
    constexpr bool full_log = false;

    if (rank == 0){
    std::cout<<"Running "<< output_fname<<" on a "<<nx<<"x"<<ny<<" grid on "<<size<<" cores today."<<std::endl;
  }

    const int my_ny = ny;
    const int my_nx = nx;
    double size_x_;
    double size_y_;
    bool reflective_;

    std::vector<double> global_h0(nx*ny);
    std::vector<double> global_hu0(nx*ny);
    std::vector<double> global_hv0(nx*ny);
    std::vector<double> global_z(nx*ny);
    std::vector<double> global_zdx(nx*ny);
    std::vector<double> global_zdy(nx*ny);


    if (rank == 0){
      SWESolver solver(test_case_id, nx, ny);
      //my_ny = solver.get_ny();
      //my_nx = solver.get_nx();
      size_x_ = solver.get_size_x();
      size_y_ = solver.get_size_y();
      reflective_ = solver.is_reflective();
      global_h0 = solver.get_h0();
      global_hu0 = solver.get_hu0();
      global_hv0 = solver.get_hv0();
      global_z = solver.get_z();
      global_zdx = solver.get_zdx();
      global_zdy = solver.get_zdy();
    }

    MPI_Bcast(&size_x_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size_y_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflective_, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(size);
    std::vector<int> displacements(size);
    get_sendcounts_displacements(my_ny, my_nx, sendcounts, displacements, size);
   
    const int local_size = sendcounts[rank]; //how much data the current core gets

    std::vector<double> temp_local_h0_(local_size); //height at time t-1
    std::vector<double> temp_local_hu0_(local_size); //x speed at time t-1
    std::vector<double> temp_local_hv0_(local_size); //y speed at time t-1
    std::vector<double> local_z_(local_size); // topography
    std::vector<double> temp_local_zdx_(local_size); //x derivative of topography
    std::vector<double> temp_local_zdy_(local_size); //y derivative of topography    

    MPI_Scatterv(global_h0.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, temp_local_h0_.data(),   local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_hu0.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, temp_local_hu0_.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_hv0.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, temp_local_hv0_.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_z.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, local_z_.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_zdx.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, temp_local_zdx_.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_zdy.data(), sendcounts.data(), displacements.data(), MPI_DOUBLE, temp_local_zdy_.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int size_while_solving = (rank==0 or rank==(size-1) )? local_size+my_nx : local_size + 2*my_nx;
    const int size_of_zd = (rank!=(size-1))? size_while_solving-my_nx : size_while_solving;

    std::vector<double> local_h0_(size_while_solving); 
    std::vector<double> local_hu0_(size_while_solving); 
    std::vector<double> local_hv0_(size_while_solving); 
    std::vector<double> local_zdx_(size_of_zd, 0.); //add a halo region in front for easier access in compute_kernel
    std::vector<double> local_zdy_(size_of_zd, 0.);

    const int start =(rank==0)? 0:my_nx;
    const int stop = (rank==0)? local_size : local_size+my_nx;

    int ii=0;
    for(int i=start; i<stop; ++i){
      local_h0_[i]=temp_local_h0_[ii];
      local_hu0_[i]=temp_local_hu0_[ii];
      local_hv0_[i]=temp_local_hv0_[ii];
      local_zdx_[i] = temp_local_zdx_[ii];
      local_zdy_[i] = temp_local_zdy_[ii];
      ++ii;
    }
    

    // Create the solver object
    ParallelSWESolver parallel_solver(my_nx, my_ny, size_x_, size_y_, reflective_, local_h0_, local_hu0_, local_hv0_, local_z_, local_zdx_, local_zdy_, rank, size);

    auto t1 = clk::now();

    parallel_solver.solve(Tend, full_log, output_n, output_fname);

    second elapsed = clk::now() - t1;
    if (j>0){
      times[j-1] = elapsed.count();
    }

    if (rank == 0 and j>0)
      std::cout << "Time for parallel SWE on iteration " << j <<" = " << times[j-1] << " [s]\n";

  } 

  if(rank == 0 and measurements > 1){
    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif
    std::cout << "Mean times for parallel SWE on "<<threads<<" threads and "<<size<<" cores = " << mean(times) << " [s]\n";
  }

  MPI_Finalize();

  return 0;
}