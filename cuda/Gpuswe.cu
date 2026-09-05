//=== GpuSWESolver.cu ===
#include "Gpuswe.hh"
#include<iostream>
#include <cmath>
#include<vector>

static __device__ inline double& at(double* vec, const std::size_t i, const std::size_t j, const std::size_t nx_){
    return vec[j*nx_ + i];
}
static __device__ inline const double& at(const double* vec, const std::size_t i, const std::size_t j, const std::size_t nx_){
    return vec[j*nx_ + i];
}
static __device__ void compute_kernel(const std::size_t i, const std::size_t j, 
                                      const double dt,
                                      const double* h0, const double* hu0, const double* hv0,
                                      double* h, double* hu, double* hv, 
                                      const std::size_t nx, const std::size_t ny, 
                                      const double size_x, const double size_y,
                                      const double* zdx, const double* zdy){

    constexpr double g = 127267.20000000;

    const double dx = size_x / nx;
    const double dy = size_y / ny;
    const double C1x = 0.5 * dt / dx;
    const double C1y = 0.5 * dt / dy;
    const double C2 = dt * g;
    constexpr double C3 = 0.5 * g;

    double hij = 0.25 * (at(h0, i, j - 1, nx) + at(h0, i, j + 1, nx) + at(h0, i - 1, j, nx) + at(h0, i + 1, j, nx))
                + C1x * (at(hu0, i - 1, j, nx) - at(hu0, i + 1, j, nx)) + C1y * (at(hv0, i, j - 1, nx) - at(hv0, i, j + 1, nx));
    if (hij < 0.0)
    {
        hij = 1.0e-5;
    }

    at(h, i, j, nx) = hij;

    if (hij > 0.0001)
    {
        at(hu, i, j, nx) =
        0.25 * (at(hu0, i, j - 1, nx) + at(hu0, i, j + 1, nx) + at(hu0, i - 1, j, nx) + at(hu0, i + 1, j, nx)) - C2 * hij * at(zdx, i, j, nx)
        + C1x
            * (at(hu0, i - 1, j, nx) * at(hu0, i - 1, j, nx) / at(h0, i - 1, j, nx) + C3 * at(h0, i - 1, j, nx) * at(h0, i - 1, j, nx)
                - at(hu0, i + 1, j, nx) * at(hu0, i + 1, j, nx) / at(h0, i + 1, j, nx) - C3 * at(h0, i + 1, j, nx) * at(h0, i + 1, j, nx))
        + C1y
            * (at(hu0, i, j - 1, nx) * at(hv0, i, j - 1, nx) / at(h0, i, j - 1, nx)
                - at(hu0, i, j + 1, nx) * at(hv0, i, j + 1, nx) / at(h0, i, j + 1, nx));

        at(hv, i, j, nx) =
        0.25 * (at(hv0, i, j - 1, nx) + at(hv0, i, j + 1, nx) + at(hv0, i - 1, j, nx) + at(hv0, i + 1, j, nx)) - C2 * hij * at(zdy, i, j, nx)
        + C1x
            * (at(hu0, i - 1, j, nx) * at(hv0, i - 1, j, nx) / at(h0, i - 1, j, nx)
                - at(hu0, i + 1, j, nx) * at(hv0, i + 1, j, nx) / at(h0, i + 1, j, nx))
        + C1y
            * (at(hv0, i, j - 1, nx) * at(hv0, i, j - 1, nx) / at(h0, i, j - 1, nx) + C3 * at(h0, i, j - 1, nx) * at(h0, i, j - 1, nx)
                - at(hv0, i, j + 1, nx) * at(hv0, i, j + 1, nx) / at(h0, i, j + 1, nx) - C3 * at(h0, i, j + 1, nx) * at(h0, i, j + 1, nx));
    }
    else
    {
        at(hu, i, j, nx) = 0.0;
        at(hv, i, j, nx) = 0.0;
    }
}
static __global__ void solve_step_kernel(const double dt,
                                         const double* h0,  const double* hu0, const double* hv0,
                                         const double* zdx, const double* zdy,
                                         double*       h1,  double*       hu1, double*       hv1,
                                         const std::size_t  nx, const std::size_t   ny,
                                         const double size_x, const double size_y){

    const std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) 
      return;

    compute_kernel(i, j, 
                   dt, 
                   h0, hu0, hv0, 
                   h1, hu1, hv1, 
                   nx, ny, 
                   size_x, size_y,
                   zdx, zdy);

  }

  __global__ void update_bcs(const double* h0, const double* hu0, const double* hv0,
                             double* h, double* hu, double* hv, 
                             std::size_t nx, std::size_t ny, 
                             const bool reflective){

      //Enough 1D-blocks should be launched so that every index is covered.

    const double coef = reflective ? -1.: 1.;
    const std::size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < nx){
        //Top and bottom boundaries
        
        const std::size_t bottom_row = nx*(ny-1);
        const std::size_t bottom_row_up_one = nx*(ny-2);

        h[tid] = h0[nx +tid];
        h[bottom_row + tid] = h0[tid+bottom_row_up_one];

        hu[tid] = hu0[nx +tid];
        hu[bottom_row + tid] = hu0[tid+bottom_row_up_one];

        hv[tid] = coef *hv0[nx +tid];
        hv[bottom_row + tid] = coef *hv0[tid+bottom_row_up_one];
      }

    if (tid < ny){
        //Left and right boundaries

        const std::size_t jnx = tid*nx;

        h[jnx] = h0[jnx +1];
        h[nx-1+jnx] = h0[jnx + nx-2];

        hu[jnx] = coef*hu0[jnx +1];
        hu[nx-1+jnx] = coef*hu0[jnx + nx-2];

        hv[jnx] = hv0[jnx +1];
        hv[nx-1+jnx] = hv0[jnx + nx-2];
        
    }
}

__global__ void compute_time_step(const double* h, const double* hu, const double* hv, 
                                  const double T, const double Tend, double* dt,
                                  const std::size_t nx, const std::size_t ny,
                                  const double size_x, const double size_y){


  if (blockIdx.x!=0 or blockIdx.y!=0)
    return;

  constexpr double g = 127267.20000000;

  double max_nu_sqr = 0.0;
  for(unsigned int j=threadIdx.y+1; j<ny-1; j+=blockDim.y){
    for(unsigned int i=threadIdx.x+1; i<nx-1; i+=blockDim.x){
      if(j<ny-1 and i<nx-1){

        const std::size_t index = j*nx + i;
        const double current_h = h[index];
        const double term = sqrt(g*current_h);

        const double nu_u = fabs(hu[index]/current_h) + term;
        const double nu_v = fabs(hv[index]/current_h) + term;

        max_nu_sqr = fmax(max_nu_sqr, nu_u*nu_u + nu_v*nu_v);
      }
    }
  }

  const unsigned int thread_id = threadIdx.y*blockDim.x + threadIdx.x;
  extern __shared__ double shared_nu_sqr[];
  shared_nu_sqr[thread_id] = max_nu_sqr;
  __syncthreads();
  
  // const unsigned int n_threads = blockDim.x*blockDim.y;
  // for(unsigned int stride=n_threads/2; stride>0; stride>>=1){
  //   if(thread_id < stride){
  //     shared_nu_sqr[thread_id] = fmax(shared_nu_sqr[thread_id], shared_nu_sqr[thread_id+stride]);
  //   }
  //   __syncthreads();
  // }
  if(thread_id==0){
      double all_max = 0.;
      for(int i=0; i<blockDim.x*blockDim.y; ++i){
        all_max = fmax(all_max, shared_nu_sqr[i]);
      }
      const double dx = size_x / (1.*nx);
      const double dy = size_y / (1.*ny);
      const double a_dt = fmin(dx, dy) / sqrt(2. * all_max);
      *dt = fmin(a_dt, Tend-T);
  }
}

void calculate_t_xy(unsigned int n, std::size_t &t_x, std::size_t &t_y) {
    //int t = 1<<n;

    if (n % 2 == 0) {
        // If n is even, t_x = t_y
        t_x = 1<<(n/2);
        t_y = t_x;
    } else {
        // If n is odd, t_x = 2 * t_y
        t_y = 1<<(n/2);
        t_x = 2 * t_y;
    }
}

//---------------------------------------------------------------------------------------------------------------------
// constructor: allocate everything once
GpuSWESolver::GpuSWESolver(std::size_t nx, std::size_t ny, double size_x, double size_y, double Tend, bool reflective){
  nx_ = nx;
  ny_ = ny;
  size_x_=size_x;
  size_y_=size_y;
  current_=0;
  Tend_=Tend;
  reflective_=reflective;

  const std::size_t N = nx_*ny_;
  const std::size_t bytes = N * sizeof(double);

  // allocate ping‐pong buffers
  for(int b=0;b<2;b++){
    cudaMalloc(&d_h[b], bytes);
    cudaMalloc(&d_hu[b], bytes);
    cudaMalloc(&d_hv[b], bytes);
  }
  cudaMalloc(&d_zdx, bytes);
  cudaMalloc(&d_zdy, bytes);
  cudaMallocManaged(&d_dt, sizeof(double));
  *d_dt = 0.;

  // build launch configurations
  constexpr unsigned int power = 0;
  constexpr int threads = 1<<power;
  std::size_t threads_for_block_x, threads_for_block_y;
  calculate_t_xy(power, threads_for_block_x, threads_for_block_y);

  this->block_kernel = dim3(threads_for_block_x, threads_for_block_y);
  this->grid_kernel  = dim3((nx_ + block_kernel.x - 1)/block_kernel.x, (ny_ + block_kernel.y - 1)/block_kernel.y);

  const std::size_t M = std::max(nx, ny);
  this->block_bcs = dim3(threads);
  this->grid_bcs = dim3((M+threads-1)/threads);

  this->block_dt = dim3(threads_for_block_x, threads_for_block_y);

  std::cout<<"Running with "<<threads<<" threads today."<<std::endl;
}

//————————————————————————————————————————————
// one GPU time‐step, ping‐pong buffers
void GpuSWESolver::step(const double& T)
{
  const int old = current_;
  const int next = 1-current_;

  

  compute_time_step<<<1, block_dt, (block_dt.x+1)*block_dt.y*sizeof(double)>>>(d_h[old], d_hu[old], d_hv[old],
                                                                           T, Tend_, d_dt,
                                                                           nx_, ny_,
                                                                           size_x_, size_y_);
  cudaDeviceSynchronize();

  dt = *d_dt;
  //std::cout<<"dt computed in gpu step = " << dt << std::endl;

  update_bcs<<<grid_bcs, block_bcs>>>(d_h[old], d_hu[old], d_hv[old],
                                      d_h[next], d_hu[next], d_hv[next],
                                      nx_, ny_, 
                                      reflective_);


  


  
  solve_step_kernel<<<grid_kernel, block_kernel>>>(dt,
                                                   d_h [old], d_hu[old], d_hv[old],
                                                   d_zdx, d_zdy,
                                                   d_h[next], d_hu[next], d_hv[next],
                                                   nx_, ny_, 
                                                   size_x_, size_y_);
  cudaDeviceSynchronize();
  
  current_ = next;
}

//————————————————————————————————————————————
// destructor: free
GpuSWESolver::~GpuSWESolver(){
  for(int b=0;b<2;b++){
    cudaFree(d_h [b]);
    cudaFree(d_hu[b]);
    cudaFree(d_hv[b]);
  }
  cudaFree(d_zdx);
  cudaFree(d_zdy);
  cudaFree(d_dt);
}

//————————————————————————————————————————————
// one‐time host -> device
void GpuSWESolver::upload_initial(
  const std::vector<double>& h0,
  const std::vector<double>& hu0,
  const std::vector<double>& hv0, 
  const std::vector<double>& zdx,
  const std::vector<double>& zdy)
{
  const std::size_t bytes = nx_*ny_*sizeof(double);
  cudaMemcpy(d_h[0], h0.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hu[0], hu0.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hv[0], hv0.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_zdx, zdx.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_zdy, zdy.data(), bytes, cudaMemcpyHostToDevice);
}

//————————————————————————————————————————————
//device -> host
void GpuSWESolver::download(std::vector<double>& h) const{
  const std::size_t bytes = nx_*ny_*sizeof(double);
  cudaMemcpy(h.data(), d_h[current_], bytes, cudaMemcpyDeviceToHost);
}

void GpuSWESolver::download_all(std::vector<double>& h, std::vector<double>& hu, std::vector<double>& hv, const int index) const{
  const std::size_t bytes = nx_*ny_*sizeof(double);
  cudaMemcpy(h.data(), d_h[index], bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hu.data(), d_hu[index], bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hv.data(), d_hv[index], bytes, cudaMemcpyDeviceToHost);
}
