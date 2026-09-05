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
#include <chrono>
using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

void ParallelSWESolver::halo_exchange(const double* vec, int tag_base, 
                                       MPI_Comm comm, MPI_Request send_reqs[2], MPI_Request recv_reqs[2],
                                       double* recv_left, double* recv_right) const {
    /*
    Communicate boundary rows except for first row of process 0 and last row of process n-1.
    */
    MPI_Irecv(recv_left, nx_, MPI_DOUBLE, prev_prank_, tag_base+1, comm, &recv_reqs[0]);
    MPI_Irecv(recv_right, nx_, MPI_DOUBLE, next_prank_, tag_base, comm, &recv_reqs[1]);

    MPI_Isend(vec, nx_, MPI_DOUBLE, prev_prank_, tag_base, comm, &send_reqs[0]);
    MPI_Isend(vec + (local_ny_-1)*nx_, nx_, MPI_DOUBLE, next_prank_, tag_base+1, comm, &send_reqs[1]);

    /*
    next_prank_ = last_rank ? MPI_PROC_NULL : rank + 1;
    prev_prank_ = first_rank ? MPI_PROC_NULL : rank - 1;
    */
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
  const std::size_t end_j = (first_rank_ or last_rank_ ) ? local_ny_ : local_ny_+1; //avoid last row if rank=n-1 and take no offset into account for rank=0
    
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

    const std::size_t chop_left = first_rank_ ? 0: nx_; //remove left halo data
    const std::size_t chop_right = last_rank_ ? 0: nx_; //remove right halo data

    
    std::shared_ptr<XDMFWriter> writer;
    
    if (output_n > 0){
        std::vector<double> global_z(this->nx_ * this-> ny_); //vector to write to file
        std::vector<double>& local_z = this->z_;
        std::vector<double> local_h0 = std::vector<double>(h0_.begin() + chop_left, h0_.end() - chop_right);
        
        //std::vector<double>& local_h0 = this->h0_;
        MPI_Gatherv(local_z.data(), local_z.size(), MPI_DOUBLE, global_z.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_h0.data(), local_h0.size(), MPI_DOUBLE, gathered_h.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (first_rank_){
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

    if (first_rank_)
        std::cout << "Solving SWE on rank " << rank << "..." << std::endl;

    std::size_t nt = 1;
    std::size_t print_dt = 0;

    std::vector<double> gathered_h1_(nx_ * ny_); //data to write to file

    const std::size_t count_send = nx_;
    std::vector<double> recvh0_left(count_send), recvh0_right(count_send),
                        recvhu0_left(count_send), recvhu0_right(count_send),
                        recvhv0_left(count_send), recvhv0_right(count_send);

    MPI_Request send[2], recv[2];

    const std::size_t N = h0.size();

    std::vector<double> time_steps;
    std::vector<double> vt_bcs;
    std::vector<double> vt_kernel;
    std::vector<double> iter_times;

    const std::size_t nx2=2*nx_;
    const int nx3=3*nx_;

    std::vector<double> send_all_left(nx3);
    std::vector<double> recv_all_left(nx3);

    std::vector<double> send_all_right(nx3);
    std::vector<double> recv_all_right(nx3);

    while (T < Tend)
    {

      auto t_iter = clk::now();

      for(std::size_t i=0; i<nx_; ++i){
        send_all_left[i]=h0[nx_+i];
        send_all_left[i+nx_] = hu0[nx_+i];
        send_all_left[i+nx2]=hv0[nx_+i];

        send_all_right[i] = h0[N-nx2+i];
        send_all_right[i+nx_]= hu0[N-nx2+i];
        send_all_right[i+nx2] = hv0[N-nx2+i];
      }

      MPI_Isend(send_all_left.data(), nx3, MPI_DOUBLE, prev_prank_, 10, MPI_COMM_WORLD, &send[0]);
      MPI_Isend(send_all_right.data(), nx3, MPI_DOUBLE, next_prank_, 20, MPI_COMM_WORLD, &send[1]);

      MPI_Irecv(recv_all_right.data(), nx3, MPI_DOUBLE, next_prank_, 10, MPI_COMM_WORLD, &recv[0]);
      MPI_Irecv(recv_all_left.data(), nx3, MPI_DOUBLE, prev_prank_, 20, MPI_COMM_WORLD, &recv[1]);

      auto t_time_step = clk::now();
      const double dt = this->compute_time_step(h0, hu0, hv0, T, Tend); //avoids halo regions
      second elapsed_time_step = clk::now()-t_time_step;
      time_steps.push_back(elapsed_time_step.count());
      
      if (first_rank_ and output_n >0 and print_dt%128 == 0){
        std::cout << "print_dt = " << print_dt << ", T = " << T << std::endl;
        std::cout << std::endl;
      }
        

        const double T1 = T + dt;
        if (first_rank_ and print_dt%128==0){
          //std::cout<<"print_dt = " << print_dt<<", dt =  " << dt <<std::endl;
          printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
          std::cout << (full_log ? "\n" : "\r") << std::flush;
        }
        ++print_dt;
        auto t_bcs = clk::now();
        this->update_bcs(h0, hu0, hv0, h, hu, hv); //avoids halo regions
        second elapsed_bcs = clk::now() - t_bcs;
        vt_bcs.push_back(elapsed_bcs.count());        
        
        auto t_kernel = clk::now();
        this->solve_step(dt, h0, hu0, hv0, h, hu, hv); //start from so that no halo regions are needed yet
        second e_ker = clk::now() - t_kernel;
        vt_kernel.push_back(e_ker.count());

        MPI_Waitall(2, recv, MPI_STATUSES_IGNORE);

        if(not first_rank_){
          for(std::size_t i=0; i<nx_; ++i){
            h0[i]=recv_all_left[i];
            hu0[i]=recv_all_left[i+nx_];
            hv0[i]=recv_all_left[i+nx2];
          }
        }
        if(not last_rank_){
          for(std::size_t i=0; i<nx_; ++i){
            h0[N-nx_+i]=recv_all_right[i];
            hu0[N-nx_+i]=recv_all_right[i+nx_];
            hv0[N-nx_+i]=recv_all_right[i+nx2];
          }
        }

        for (std::size_t i = 1; i < nx_ - 1; ++i){
          compute_kernel(i, 1, dt, h0, hu0, hv0, h, hu, hv);
        }
        for (std::size_t i = 1; i < nx_ - 1; ++i){
          compute_kernel(i, stop_y-1, dt, h0, hu0, hv0, h, hu, hv);
        }
        
        if (output_n > 0 && nt % output_n == 0){
          std::vector<double> to_gather = std::vector<double>(h.begin() + chop_left, h.end() - chop_right);
          MPI_Gatherv(h.data(), h.size(), MPI_DOUBLE, gathered_h.data(), my_sendcounts.data(), my_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
          if (rank == 0){
              writer->add_h(gathered_h, T1);
          }
        }
        ++nt;

        MPI_Waitall(2, send, MPI_STATUSES_IGNORE);

        // Swap the old and new solutions
        std::swap(h, h0);
        std::swap(hu, hu0);
        std::swap(hv, hv0);

        T = T1;
        second iter_timesss = clk::now()-t_iter;
        iter_times.push_back(iter_timesss.count());

        if (print_dt>100 and size==1){
          if (first_rank_){
            std::cout<<"Going out of for loop after one iteration. Remove if you're not debugging."<<std::endl;
          }
          break;
        }
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
        if(first_rank_){
            writer->add_h(gathered_h1_, T);
        }
    }
    if (first_rank_){
      std::cout << "Finished solving SWE afer "<<print_dt<<" iterations." << std::endl << std::endl;
      const double m_ts = std::accumulate(time_steps.begin(), time_steps.end(), 0.) / (1.*time_steps.size());
      const double m_bcs = std::accumulate(vt_bcs.begin(), vt_bcs.end(), 0.) / (1.*vt_bcs.size());
      const double m_ker = std::accumulate(vt_kernel.begin(), vt_kernel.end(), 0.) / (1.*vt_kernel.size());

      std::cout<<"Mean for time step = "<< m_ts <<"[s], for bcs = "<<m_bcs<<"[s], for kernel = "<<m_ker<<"[s]."<<std::endl;
      std::cout<<"Mean for iteration = " <<std::accumulate(iter_times.begin(), iter_times.end(), 0.)/(1.*iter_times.size())<<std::endl;
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
  if(first_rank_){
    for (std::size_t i = 0; i < nx_; ++i){
      //const std::size_t bottom_row = i;
      const std::size_t bottom_row_up_one = nx_+i;
      h[i] = h0[bottom_row_up_one];
      hu[i] = hu0[bottom_row_up_one];
      hv[i] = coef * hv0[bottom_row_up_one];
    }
  }
  //Top boundaries
  if(last_rank_){
    for (std::size_t i = 0; i<nx_; ++i){
      //local_ny_ = ny_/size
      at(h, i, local_ny_ ) = at(h0, i, local_ny_ - 1);
      at(hu, i, local_ny_) = at(hu0, i, local_ny_ - 1);
      at(hv, i, local_ny_) = coef * at(hv0, i, local_ny_ - 1);
    }
  }

  // Left and right boundaries.
  const std::size_t bottom_row = first_rank_? 0 : 1;
  const std::size_t top_row = first_rank_? local_ny_ : local_ny_+1;
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
//#pragma omp parallel for 
  for (std::size_t j=2; j<stop_y-1; ++j){
    for (std::size_t i = 1; i < nx_ - 1; ++i){
      compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }
}
