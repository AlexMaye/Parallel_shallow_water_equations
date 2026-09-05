#pragma once
#include <vector>
#include <string>
#include "swe.hh"
#include <mpi.h>
#include<iostream>

class ParallelSWESolver : public SWESolver{

  public:
  
    ParallelSWESolver(std::size_t my_nx_, std::size_t my_ny_, double local_size_x_, double local_size_y_, bool local_reflective_,
                      std::vector<double>& local_h0_, std::vector<double>& local_hu0_,
                      std::vector<double>& local_hv0_,
                      std::vector<double>& local_z_, std::vector<double>& local_zdx_,  
                      std::vector<double>& local_zdy_, const int my_rank, const int my_size)
      {
        this->nx_ = my_nx_;
        this->ny_ = my_ny_;
        this->size_y_ = local_size_y_;
        this->size_x_ = local_size_x_;
        this->reflective_ = local_reflective_;
        this->h0_ = local_h0_;
        this->hu0_ = local_hu0_;
        this->hv0_ = local_hv0_;
        this->z_ = local_z_;
        this->zdx_ = local_zdx_;
        this->zdy_ = local_zdy_;
        this->rank = my_rank;
        this->size = my_size;

        const int remainder = ny_ % size;

        this->local_ny_ = (rank<remainder)? ny_/size +1 : ny_/size; //height of region without halo regions

        this->h1_ = std::vector<double>(local_h0_.size(), 0.);
        this->hu1_ = std::vector<double>(local_hu0_.size(), 0.);
        this->hv1_ = std::vector<double>(local_hv0_.size(), 0.);

        last_rank_ = rank==(size-1);
        first_rank_ = rank==0;

        //We modify h which has not received any halo regions by accessing elements in h0 which has received halo regions. The indices we pass to 
        //compute kernel are those of h we want to modify. Therefore,
        //for rank 0, we modify h from row 1 to its last row included,
        //for rank i, we modify h from row 0 to its last row included,
        //for last rank, we modify h from row 0 to its last row excluded.

        this->start_y = first_rank_? 1:0; 
        this->stop_y = (last_rank_ or first_rank_)? local_ny_ : local_ny_+1; //where to stop iterating.

        
        next_prank_ = last_rank_ ? MPI_PROC_NULL : rank + 1;
        prev_prank_ = first_rank_ ? MPI_PROC_NULL : rank - 1;

        const int added_height = (first_rank_ or last_rank_)? 1:2;

        vec_size_ = nx_ * (local_ny_+added_height);

      }
  
    void solve(const double Tend, const bool full_log = false, const std::size_t output_n = 0, const std::string &fname_prefix = "test") override;

  protected:
    double compute_time_step(const std::vector<double> &h, const std::vector<double> &hu, const std::vector<double> &hv, const double T,
                             const double Tend) const override;

    void update_bcs(const std::vector<double> &h0, const std::vector<double> &hu0, const std::vector<double> &hv0, std::vector<double> &h, 
                    std::vector<double> &hu, std::vector<double> &hv) const override;

    void solve_step(const double dt,
                  std::vector<double> &h0,
                  std::vector<double> &hu0,
                  std::vector<double> &hv0,
                  std::vector<double> &h,
                  std::vector<double> &hu,
                  std::vector<double> &hv) override;

    public:

    void set_sendcounts(const std::vector<int>& sendcounts_){
        sendcounts = sendcounts_;
    }
    void set_displacements(const std::vector<int>& displacements_){
        displacements = displacements_;
    }
  
    private:

    void halo_exchange(const double* vec, int tag_base, 
                                       MPI_Comm comm, MPI_Request send_reqs[2], MPI_Request recv_reqs[2],
                                       double* recv_left, double* recv_right) const;
    int rank;
    int size;
    std::size_t local_ny_;
    std::vector<int> sendcounts;
    std::vector<int> displacements;
    std::size_t start_y;
    std::size_t stop_y;
    int next_prank_;
    int prev_prank_;
    bool last_rank_;
    bool first_rank_;
    int vec_size_;
  
  };