#include "parallel_swe.hh"
#include <mpi.h>
#include "utils.cpp"
#include "xdmf_writer.hh"

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>

#include<numeric>
#include<chrono>
using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

void ParallelSWESolver::halo_send_recv(std::vector<double>& my_vec){
    /*
    Communicate boundary rows except for first row of process 0 and last row of process n-1.
    */
    const std::size_t count_send = this->nx_;
    const bool last_rank = rank == (size-1); //is true if rank == size-1
    const bool first_rank = rank == 0; //is true if rank == 0
    const int next_prank = last_rank ? MPI_PROC_NULL : rank + 1;
    const int prev_prank = first_rank ? MPI_PROC_NULL : rank - 1;

    std::vector<double> concat_after_my_vec(count_send);
    std::vector<double> concat_before_my_vec(count_send);

    const std::size_t chop_left = (rank!=0) ? nx_ : 0; //remove left halo data
    const std::size_t chop_right = (rank!=size-1) ? nx_ : 0; //remove right halo data
    
    const std::vector<double> my_vec_last_elems = std::vector<double>(my_vec.end()-count_send-chop_right, my_vec.end()-chop_right);
    const std::vector<double> my_vec_first_elems = std::vector<double>(my_vec.begin()+chop_left, my_vec.begin()+chop_left+count_send);

    MPI_Sendrecv(my_vec_first_elems.data(), count_send, MPI_DOUBLE, prev_prank, 0, concat_after_my_vec.data(), count_send, MPI_DOUBLE, next_prank, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    MPI_Sendrecv(my_vec_last_elems.data(), count_send, MPI_DOUBLE, next_prank, 1, concat_before_my_vec.data(), count_send, MPI_DOUBLE, prev_prank, 1, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
    
    if (not first_rank){
      for(std::size_t i=0; i<nx_; ++i){
        my_vec[i] = concat_before_my_vec[i];
      }
    }
    if (not last_rank){
      const std::size_t N = my_vec.size();
      std::size_t j=0;
      for(std::size_t i=N-nx_; i<N; ++i){
        my_vec[i] = concat_after_my_vec[j];
        ++j;
      }
    }
}

double ParallelSWESolver::compute_time_step(const std::vector<double> &h,
                             const std::vector<double> &hu,
                             const std::vector<double> &hv,
                             const double T,
                             const double Tend) const
{
    // Calculate local maximum values
  double local_max_nu_sqr = 0.0;
  //double local_au{0.0};
  //double local_av{0.0};
  
  // Adjust loop bounds based on process position. 
  constexpr std::size_t start_j = 1; //avoid first row if rank=0 and halo region otherwise
  const std::size_t end_j = (rank==0 or rank==(size-1) ) ? local_ny_ : local_ny_+1; //avoid last row if rank=n-1
  
  //std::cout<<"Rank "<<rank<<" going from "<<start_j<<" to "<<end_j<<std::endl;
  
  for (std::size_t j = start_j; j < end_j; ++j){
    for (std::size_t i = 1; i < nx_ - 1; ++i){
      //local_au = std::max(local_au, std::fabs(at(hu, i, j)));
      //local_av = std::max(local_av, std::fabs(at(hv, i, j)));
      const std::size_t index = i+j*nx_;
      const double current_h = h[index];
      const double term = sqrt(g*current_h);
      const double nu_u = std::fabs(hu[index]) / current_h + term;
      const double nu_v = std::fabs(hv[index]) / current_h + term;
      local_max_nu_sqr = std::max(local_max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    }
  }

  // Find global maximum using MPI_Allreduce
  double global_max_nu_sqr = 0.0;
  MPI_Allreduce(&local_max_nu_sqr, &global_max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
  // We still use global grid dimensions for dx and dy calculation
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;  // Using global ny_
  
  double dt = std::min(dx, dy) / (sqrt(2.0 * global_max_nu_sqr));
  return std::min(dt, Tend - T);
}

void 
ParallelSWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix){
    
    const std::vector<int>& my_sendcounts = this->sendcounts;
    const std::vector<int>& my_displacements = this->displacements;
    std::vector<double> gathered_h(nx_ * ny_); //vector to write to file

    const std::size_t chop_left = (rank!=0) ? nx_ : 0; //remove left halo data
    const std::size_t chop_right = (rank!=size-1) ? nx_ : 0; //remove right halo data
    
    std::shared_ptr<XDMFWriter> writer;
    
    if (output_n > 0){
        std::vector<double> global_z(this->nx_ * this-> ny_); //vector to write to file
        std::vector<double>& local_z = this->z_;
        std::vector<double> local_h0 = std::vector<double>(h0_.begin() + chop_left, h0_.end() - chop_right);
        
        //std::vector<double>& local_h0 = this->h0_;
        MPI_Gatherv(local_z.data(), local_z.size(), MPI_DOUBLE, global_z.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_h0.data(), local_h0.size(), MPI_DOUBLE, gathered_h.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0){
            writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, global_z);
            writer->add_h(gathered_h, 0.0);
        }
    }

    double T = 0.0;

    std::vector<double> &h = h1_;
    std::vector<double> &hu = hu1_;
    std::vector<double> &hv = hv1_;

    std::vector<double> &h0 = h0_;
    std::vector<double> &hu0 = hu0_;
    std::vector<double> &hv0 = hv0_;

    if (rank == 0)
        std::cout << "Solving SWE on rank " << rank << "..." << std::endl;

    std::size_t nt = 1;
    std::size_t print_dt = 0;

    std::vector<double> gathered_h1_(nx_ * ny_); //data to write to file

    std::vector<double> iter_times;

    while (T < Tend)
    {
      auto t_time_step = clk::now();
        const double dt = this->compute_time_step(h0, hu0, hv0, T, Tend);
        if (print_dt % 256 == 0 and rank == 0 and output_n >0){
          std::cout << "print_dt = " << print_dt << ", dt = " << dt << std::endl;
          std::cout << std::endl;
        }
        

        const double T1 = T + dt;
        if (rank == 0 and print_dt%256==0){
          //std::cout<<"print_dt = " << print_dt<<", dt =  " << dt <<std::endl;
          printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
          std::cout << (full_log ? "\n" : "\r") << std::flush;
        }
        ++print_dt;
        this->update_bcs(h0, hu0, hv0, h, hu, hv);

        this->solve_step(dt, h0, hu0, hv0, h, hu, hv);

        if (output_n > 0 && nt % output_n == 0){
          std::vector<double> to_gather = std::vector<double>(h.begin() + chop_left, h.end() - chop_right);
          MPI_Gatherv(h.data(), h.size(), MPI_DOUBLE, gathered_h.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
          if (rank == 0){
              writer->add_h(gathered_h, T1);
          }
        }
        ++nt;

        // Swap the old and new solutions
        std::swap(h, h0);
        std::swap(hu, hu0);
        std::swap(hv, hv0);

        T = T1;
        if (size==1 and print_dt>120){
          if (rank==0){
            std::cout<<"Going out of for loop after one iteration. Remove if you're not debugging."<<std::endl;
          }
          break;
        }

        second elapsed = clk::now() - t_time_step;
        iter_times.push_back(elapsed.count());
    }

    // Copying last computed values to h1_, hu1_, hv1_ (if needed)
    if (&h0 != &h1_)
    {
        h1_ = h0;
        hu1_ = hu0;
        hv1_ = hv0;
    }

    if (output_n > 0)
    {
        MPI_Gatherv(h1_.data(), h1_.size(), MPI_DOUBLE, gathered_h1_.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank == 0){
            writer->add_h(gathered_h1_, T);
        }
    }
    if (rank == 0){
      std::cout << "Finished solving SWE after " << print_dt<<" iterations." << std::endl << std::endl;
      std::cout<< "Mean per time step = " << std::accumulate(iter_times.begin(), iter_times.end(), 0.) / (1.*iter_times.size())<<std::endl;
    }
}

void
ParallelSWESolver::update_bcs(const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const
{

  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Bottom boundaries.
  if(rank == 0){
    for (std::size_t i = 0; i < nx_; ++i){
      at(h, i, 0) = at(h0, i, 1);
      at(hu, i, 0) = at(hu0, i, 1);
      at(hv, i, 0) = coef * at(hv0, i, 1);
    }
  }
  //Top boundaries
  if(rank == size-1){
    for (std::size_t i = 0; i<nx_; ++i){
      //local_ny_ = ny_/size
      at(h, i, local_ny_ ) = at(h0, i, local_ny_ - 1);
      at(hu, i, local_ny_) = at(hu0, i, local_ny_ - 1);
      at(hv, i, local_ny_) = coef * at(hv0, i, local_ny_ - 1);
    }
  }

  // Left and right boundaries.
  const std::size_t bottom_row = (rank==0)? 0 : 1;
  const std::size_t top_row = (rank==0)? local_ny_ : local_ny_+1;
  for (std::size_t j = bottom_row; j < top_row; ++j){
    
    at(h, 0, j) = at(h0, 1, j); //left boundary (column 0, row j)
    at(h, nx_ - 1, j) = at(h0, nx_ - 2, j); //right boundary (column nx_-1, row j)

    at(hu, 0, j) = coef * at(hu0, 1, j);
    at(hu, nx_ - 1, j) = coef * at(hu0, nx_ - 2, j);

    at(hv, 0, j) = at(hv0, 1, j);
    at(hv, nx_ - 1, j) = at(hv0, nx_ - 2, j);
  }

};

#ifdef _OPENMP
 #include <omp.h>
#endif
//#include <cstdio>
void
ParallelSWESolver::solve_step(const double dt,
                      std::vector<double> &h0,
                      std::vector<double> &hu0,
                      std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv)
{

  halo_send_recv(h0);
  halo_send_recv(hu0); //BE CAREFUL NOT TO COMMENT THIS OUT
  halo_send_recv(hv0);

  //#pragma omp parallel for
  for (std::size_t j=start_y; j<stop_y; ++j){
    for (std::size_t i = 1; i < nx_ - 1; ++i){
      compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }

}

void
ParallelSWESolver::compute_kernel(const std::size_t i,
  const std::size_t j,
  const double dt,
  const std::vector<double> &h0,
  const std::vector<double> &hu0,
  const std::vector<double> &hv0,
  std::vector<double> &h,
  std::vector<double> &hu,
  std::vector<double> &hv) const{

    const double dx = size_x_ / nx_;
    const double dy = size_y_ / ny_;
    const double C1x = 0.5 * dt / dx;
    const double C1y = 0.5 * dt / dy;
    const double C2 = dt * g;
    constexpr double C3 = 0.5 * g;

    const std::size_t j0 = (rank==0)? j : j+1;
    

    double hij = 0.25 * (at(h0, i, j0 - 1) + at(h0, i, j0 + 1) + at(h0, i - 1, j0) + at(h0, i + 1, j0))
                + C1x * (at(hu0, i - 1, j0) - at(hu0, i + 1, j0)) + C1y * (at(hv0, i, j0 - 1) - at(hv0, i, j0 + 1));
    if (hij < 0.0)
    {
      hij = 1.0e-5;
    }

    at(h, i, j0) = hij;

    if (hij > 0.0001)
    {
      at(hu, i, j0) =
        0.25 * (at(hu0, i, j0 - 1) + at(hu0, i, j0 + 1) + at(hu0, i - 1, j0) + at(hu0, i + 1, j0)) - C2 * hij * at(zdx_, i, j)
        + C1x
            * (at(hu0, i - 1, j0) * at(hu0, i - 1, j0) / at(h0, i - 1, j0) + C3 * at(h0, i - 1, j0) * at(h0, i - 1, j0)
              - at(hu0, i + 1, j0) * at(hu0, i + 1, j0) / at(h0, i + 1, j0) - C3 * at(h0, i + 1, j0) * at(h0, i + 1, j0))
        + C1y
            * (at(hu0, i, j0 - 1) * at(hv0, i, j0 - 1) / at(h0, i, j0 - 1)
              - at(hu0, i, j0 + 1) * at(hv0, i, j0 + 1) / at(h0, i, j0 + 1));

      at(hv, i, j0) =
        0.25 * (at(hv0, i, j0 - 1) + at(hv0, i, j0 + 1) + at(hv0, i - 1, j0) + at(hv0, i + 1, j0)) - C2 * hij * at(zdy_, i, j)
        + C1x
            * (at(hu0, i - 1, j0) * at(hv0, i - 1, j0) / at(h0, i - 1, j0)
              - at(hu0, i + 1, j0) * at(hv0, i + 1, j0) / at(h0, i + 1, j0))
        + C1y
            * (at(hv0, i, j0 - 1) * at(hv0, i, j0 - 1) / at(h0, i, j0 - 1) + C3 * at(h0, i, j0 - 1) * at(h0, i, j0 - 1)
              - at(hv0, i, j0 + 1) * at(hv0, i, j0 + 1) / at(h0, i, j0 + 1) - C3 * at(h0, i, j0 + 1) * at(h0, i, j0 + 1));
    }
    else
    {
      at(hu, i, j0) = 0.0;
      at(hv, i, j0) = 0.0;
    }

  }
