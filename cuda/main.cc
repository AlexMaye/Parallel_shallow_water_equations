#include "swe.hh"

#include <string>
#include <cstddef>

#include <chrono>
#include <iostream>
#include <numeric>
using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

template<typename T>
double mean(std::vector<T> a){
  double sum = std::accumulate(a.begin(), a.end(), 0.);
  return sum/a.size();
}

int main(){
  constexpr int number_of_runs = 3;
  std::vector<double> times(number_of_runs-1, 0.);
  
  // Uncomment the option you want to run.

  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = 1;  // Water drops in a box
  const double Tend = 1.0;     // Simulation time in hours
  constexpr std::size_t nx = 300; // Number of cells per direction.
  constexpr std::size_t ny = 300; // Number of cells per direction.
  constexpr std::size_t output_n = 0;
  const std::string output_fname = "water_drops";
  const bool full_log = false;
  for (int i = 0; i<number_of_runs; ++i){

    // {
    //   SWESolver solver(test_case_id, nx, ny);
    //   solver.solve(Tend, full_log, output_n, output_fname);
    // }
    {
      auto t1 = clk::now();
      SWESolver solver(test_case_id, nx, ny);
      solver.solve_gpu(Tend, full_log, output_n, output_fname);
      second elapsed = clk::now() - t1;
      if (i>0)
        times[i-1] = elapsed.count();
      std::cout << "Time for CGPU  = " << elapsed.count() << " [s]\n";
    }
  }

  if (number_of_runs > 1){
    std::cout << "Mean times for SWE on GPU = " << mean(times) << " [s]\n";
  }

  // // Option 2 - Solving analytical (dummy) tsunami example.
  // const int test_case_id = 2;  // Analytical tsunami test case
  // const double Tend = 1.0;     // Simulation time in hours
  // const std::size_t nx = 1000; // Number of cells per direction.
  // const std::size_t ny = 1000; // Number of cells per direction.
  // const std::size_t output_n = 10;
  // const std::string output_fname = "analytical_tsunami";
  // const bool full_log = false;

  // SWESolver solver(test_case_id, nx, ny);
  // solver.solve(Tend, full_log, output_n, output_fname);

  // // Option 3 - Solving tsunami problem with data loaded from file.
  // const double Tend = 0.2;   // Simulation time in hours
  // const double size = 500.0; // Size of the domain in km

  // // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
  // const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
  // // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
  // // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
  // // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

  // const std::size_t output_n = 0;
  // const std::string output_fname = "tsunami";
  // const bool full_log = false;

  // SWESolver solver(fname, size, size);
  // solver.solve(Tend, full_log, output_n, output_fname);

  return 0;
}
