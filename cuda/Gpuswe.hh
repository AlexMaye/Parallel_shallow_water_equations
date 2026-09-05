//=== GpuSWESolver.hh ===
#pragma once
#include <vector>
#include <cuda_runtime.h>

class GpuSWESolver {
public:
  GpuSWESolver(std::size_t nx, std::size_t ny, double size_x, double size_y, double Tend, bool reflective);
  ~GpuSWESolver();

  // copy host data → device (one‐off)
  void upload_initial(
    const std::vector<double>& h0,
    const std::vector<double>& hu0,
    const std::vector<double>& hv0,
    const std::vector<double>& zdx_,
    const std::vector<double>& zdy_);

  // advance one time‐step entirely on the GPU
  void step(const double& T);

  //copy device -> host
  void download(std::vector<double>& h) const;
  void download_all(std::vector<double>& h, std::vector<double>& hu, std::vector<double>& hv, const int index) const;

  void download_dt(double& h_dt){
        cudaMemcpy(&h_dt, d_dt, sizeof(double), cudaMemcpyDeviceToHost);
    }

    double dt;

private:
  std::size_t nx_, ny_;
  double size_x_, size_y_;
  double Tend_;
  double reflective_;
  

  // device buffers: we’ll ping‐pong between 0 and 1
  double *d_h[2], *d_hu[2], *d_hv[2];
  double *d_dt;
  int current_;      // which buffer holds “old” data

  // (device)
  double *d_zdx, *d_zdy;

  // launch‐configuration
  dim3 block_kernel, grid_kernel, block_bcs, grid_bcs, block_dt;

  // __global__ and __device__ methods live in the .cu
};
