#ifndef MULTIPHASE_LBM_H_
#define MULTIPHASE_LBM_H_
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "csv.h"
#include "lbm_model.cuh"

//! MultiphaseLBM functions and data structures
struct MultiphaseLBM {

  // The number of cluser at current node(on CPU)
  Size_txyz* liq_cluster_number;
  Boolxyz* liq_cluster;
  Size_txyz* gas_cluster_number;
  Boolxyz* gas_cluster;
  Size_txyz* solid_number;

  // Density on GPU
  Realxyz* drho;
  // Pseudopotential on GPU
  Realxyz* dpsi;
  // Pressure on GPU
  Realxyz* dpressure;
  // Distributions in collision on GPU
  RealxyzQ* dcollision_f;
  // Distributions in streaming on GPU
  RealxyzQ* dstream_f;
  // Force on GPU
  Realxyz3* dforce;
  Boolxyz* dsolid_touch_liquid;
  Boolxyz* dis_fluid;

  bool big_endian;

  bool save_vtk_ascii;
  bool save_vtr_binary;
  bool save_tecplot_ascii;

  // size_t lx;
  // size_t ly;
  // size_t lz;

  real res;

  int wanted_num_threads;

  real rho0;
  real rhos;
  real IniPerturbRate;
  real G;

  // If true Carnahan-Starling EOS is used, else Shan-Chen EOS
  bool carn_star;
  // T over Tc ratio
  real T_Tc;
  // rho used to identify the liquid nodes
  real rhol_spinodal;
  // rho used to identify the vapor nodes
  real rhog_spinodal;

  bool inject_density;
  unsigned int inject_type;
  real inject_position;
  size_t rho_inject_period;
  real rho_increment;

  bool read_densities;
  bool hysteresis;
  bool erosion;
  real max_saturation;
  real min_saturation;
  real bound_difference;
  real target_density;

  // Relaxation time
  real tau;
  // Maximum number of iterations
  long step_max;

  // Iteration counter
  // The step number
  long step;
  // Period (in number of steps) between vtk-files
  size_t vtkPeriod;
  // Period (in number of steps) between vtr-files
  size_t vtrPeriod;
  // Period (in number of steps) between tecplot-files
  size_t tecplotPeriod;
  // Period (in number of steps) between records of pressure profile
  size_t pressureProfilePeriod;
  // Period (in number of steps) between records in the flow file
  size_t statPeriod;

  std::string result_folder;
  // The stream file for storing data during the flow
  std::ofstream statFile;

  real packing_density;
  char densities_filename[256];

  size_t solid_count = 0;
  size_t liquid_count = 0;
  real wet_coord_num = 0;
  real cluster_order = 0;
  size_t solids = 0;
  size_t solid_liquid_num = 0;
  real ratio_liquid_solid_boundary;
  real ratio_liquid_vapor_boundary;
  real max_r;

  // unified memory to write stas
  real average_liquid_pressure = 0.;
  real average_vapor_pressure = 0.;
  real saturation_ratio = 0.;
  real average_liquid_density = 0.;
  real average_vapor_density = 0.;

  enum class ClusterType { liq, gas };
  size_t liq_cluster_count = 0;
  size_t gas_cluster_count = 0;
  size_t cluster_size = 0;
  bool cluster_remove = false;
  std::map<size_t, std::set<size_t>> cluster_to_solids_map;
  std::map<size_t, std::set<size_t>> solid_to_clusters_map;

  //! Constructor for initializing the parameters
  MultiphaseLBM();
  ~MultiphaseLBM();

  //! Allocate memory for the simulation on host and device
  void resize_gpu();

  //! \brief Perfrom a depth-first search to find a cluster
  //! \param[in] x x-coordinate of the first node of the cluster
  //! \param[in] y y-coordinate of the first node of the cluster
  //! \param[in] z z-coordinate of the first node of the cluster
  void depth_first_search(size_t x, size_t y, size_t z, ClusterType type);

  void calc_cluster();

  //! Run iterations on gpu
  void run_gpu();

  //! Initialize the simulation (by openning output file)
  void init();

  //! Read an input file with the physical parameters and
  // the description of the system. See MultiphaseLBM documentation for detailed
  // description of the different tokens.
  //! \param[in] name Name of the input file
  void read_data(const char* name);

  //! Read the 3D-matrix 'sample' from an ASCII file
  void read_sample(const char* name);

  //! Create a vtk ascii file (readable by Paraview) named resultXXXXXX.vtk
  void write_vtk_ascii(size_t num);

  //! Define the fluid-solid boundary inside the sample and at the domain
  // boundaries
  void find_solid_touch_liquid();
  //！Erode the fluid-solid boundary inside the sample and at the domain
  // boundaries to decrease packing density
  void erode_solid_touch_liquid();

  //! Place a single solid sphere in the domain
  //! \param[in] x x-coordinate of center of the sphere
  //! \param[in] y y-coordinate of center of the sphere
  //! \param[in] z z-coordinate of center of the sphere
  //! \param[in] R radius of the sphere
  void place_sphere(real x, real y, real z, real R);

  //! Create a fkiw file named Stats.txt
  void write_stats();

  //! Generate uniformly distirbured particles
  //! \param[in] num Number of particles to generate
  //! \param[in] rmin Minimum radius for the particle size distribution
  //! \param[in] rmax Maximum radius for the particle size distribution
  //! \param[in] xmin Lower bound in the x direction of the box in which the
  /// particles are generated
  //! \param[in] xmax Upper bound in the x direction of the box in which the
  /// particles are generated
  //! \param[in] ymin Lower bound in the y direction of the box in which the
  /// particles are generated
  //! \param[in] ymax Upper bound in the y direction of the box in which the
  /// particles are generated
  //! \param[in] zmin Lower bound in the z direction of the box in which the
  /// particles are generated
  //! \param[in] zmax Upper bound in the z direction of the box in which the
  /// particles are generated
  void generate_solids(int num, real rmin, real rmax, real xmin, real xmax,
                       real ymin, real ymax, real zmin, real zmax);

  //! Read position of spherical particles from file
  //! \param[in] name Name of file
  void read_positions(const char* name);

  //! Calculate the packing density of the specimen
  void calc_packing_density();
};

#endif  // MULTIPHASE_LBM_H_
