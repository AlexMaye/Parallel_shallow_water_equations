#pragma once

#include <cstddef>
#include <vector>
#include <string>

class SWESolver
{
protected:
  /**
   * @brief Constructor for the SWESolver class.
   * @warning Not allowed to be used.
   */
  SWESolver()=default;

public:
  /// Gravity 9.82 * (3.6)^2 * 1000 in[km / hour^2]
  static constexpr double g = 127267.20000000;

  /**
   * @brief Constructor.
   * @param test_case_id It can be 1 (water drops in a box) or 2 (analytical tsunami).
   * @param nx  Number of cells along the x direction.
   * @param ny  Number of cells along the y direction.
   */
  SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny);

  /**
   * @brief Constructor for the SWESolver class.
   *
   * This constructor corresponds to the case in which the initial conditions
   * and topography are read from a HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   * @param size_x  Size in km along the x direction.
   * @param size_y  Size in km along the y direction.
   */
  SWESolver(const std::string &h5_file, const double size_x, const double size_y);

  /**
   * @brief Solve the shallow water equations.
   * @brief Tend Total simulation time.
   * @brief full_log If true, the simulation will log the time step
   * and the time step size at each time step. Otherwise, only
   * the progress of the simulation will be logged.
   * @brief output_n If different from 0, the simulation will write
   * a solution each output_n time steps. E.g., if set to 10,
   * a solution file will be written each 10 steps.
   * @brief fname_prefix If @p output_n is different from 0, the generated
   * files will use this file name prefix.
   */
  virtual void solve(const double Tend,
             const bool full_log = false,
             const std::size_t output_n = 0,
             const std::string &fname_prefix = "test");

  std::size_t get_nx() const
  {
    return nx_;
  }
  std::size_t get_ny() const
  {
    return ny_;
  }
  double get_size_x() const
  {
    return size_x_;
  }
  double get_size_y() const
  {
    return size_y_;
  }
  bool is_reflective() const
  {
    return reflective_;
  }
  const std::vector<double> &get_h0() const
  {
    return h0_;
  }
  const std::vector<double> &get_hu0() const
  {
    return hu0_;
  }
  const std::vector<double> &get_hv0() const
  {
    return hv0_;
  }
  const std::vector<double> &get_z() const
  {
    return z_;
  }
  const std::vector<double> &get_zdx() const
  {
    return zdx_;
  }
  const std::vector<double> &get_zdy() const
  {
    return zdy_;
  }

protected:
  /**
   * @brief Initializes the initial conditions and topography using
   * the provided HDF5 file.
   *
   * @param h5_file HDF5 file name containing the initial conditions and topography.
   */
  void init_from_HDF5_file(const std::string &h5_file);

  /**
   * @brief Initializes the initial conditions and topography using
   * a Gaussian function.
   *
   * The water height is initialized with two separated Gaussian peaks.
   * The initial water velocity is set to zero and the topography is set to zero.
   */
  void init_gaussian();

  /**
   * @brief Initializes the initial conditions and topography using
   * a dummy tsunami function.
   */
  void init_dummy_tsunami();

  /**
   * @brief Initializes the initial conditions and topography using
   * a slope function.
   */
  void init_dummy_slope();

  /**
   * @brief Initializes the derivatives dx and dy from the topography.
   */
  void init_dx_dy();

  std::size_t nx_;
  std::size_t ny_;
  double size_x_;
  double size_y_;
  bool reflective_;
  std::vector<double> h0_; //height at time t-1
  std::vector<double> h1_; //height at time t
  std::vector<double> hu0_; //x speed at time t-1
  std::vector<double> hu1_; //x speed at time t
  std::vector<double> hv0_; //y speed at time t-1
  std::vector<double> hv1_; //y speed at time t
  std::vector<double> z_; // topography
  std::vector<double> zdx_; //x derivative of topography
  std::vector<double> zdy_; //y derivative of topography

  /**
   * @brief Accessor for 2D vector elements.
   */
  inline double &at(std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Accessor for 2D vector elements.
   * @note Constant vector version.
   */
  inline const double &at(const std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Updates the water height and velocities using the SWE kernel at a given cell.
   * @param i x index of the cell.
   * @param j y index of the cell.
   * @param dt Time step.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  void compute_kernel(const std::size_t i,
                      const std::size_t j,
                      const double dt,
                      const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const;

  /**
   * @brief Computes the time step size that satisfied the CFL condition.
   *
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   * @param T Current time.
   * @param Tend Final time.
   * @return Compute time step.
   */
  virtual double compute_time_step(const std::vector<double> &h,
                           const std::vector<double> &hu,
                           const std::vector<double> &hv,
                           const double T,
                           const double Tend) const;

  /**
   * @brief Solve one step of the SWE.
   * @param dt The time step size.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  virtual void solve_step(const double dt,
                  std::vector<double> &h0,
                  std::vector<double> &hu0,
                  std::vector<double> &hv0,
                  std::vector<double> &h,
                  std::vector<double> &hu,
                  std::vector<double> &hv);

  /**
   * @brief Update boundary conditions.
   * @note This function updates the boundary conditions for the SWE solver.
   * @param h0 The water height in the previous time step.
   * @param hu0 The x water velocity in the previous time step.
   * @param hv0 The y water velocity in the previous time step.
   * @param h The water height in the current time step.
   * @param hu The x water velocity in the current time step.
   * @param hv The y water velocity in the current time step.
   */
  virtual void update_bcs(const std::vector<double> &h0,
                  const std::vector<double> &hu0,
                  const std::vector<double> &hv0,
                  std::vector<double> &h,
                  std::vector<double> &hu,
                  std::vector<double> &hv) const;
};

