#include "iteration_host.h"
#include "lbm_model.cuh"
#include "multiphase_lbm.h"

//! Get time stamp
#define USE_UNIX 1
double get_time(void);
#if USE_UNIX
#include <sys/time.h>
#include <time.h>
double get_time(void) {
  struct timeval tv;
  double t;
  gettimeofday(&tv, (struct timezone*)0);
  t = tv.tv_sec + (double)tv.tv_usec * 1e-6;
  return t;
}
#else
#endif

//! Create folder to store statistics
void create_folder(const std::string& folder) {
  if (access(folder.c_str(), F_OK)) {
    int stat;
    stat = mkdir(folder.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP |
                                     S_IXGRP | S_IROTH | S_IXOTH);
    if (stat == -1)
      std::cout << "Cannot create the folder " << folder << std::endl;
  }
}

//! MultiphaseLBM class
//! \brief MultiphaseLBM functions and data structures
//! Set default values of some parameters defined in the command file.
MultiphaseLBM::MultiphaseLBM() {
  short int word = 0x0001;
  char* byte = (char*)&word;
  big_endian = (byte[0] ? false : true);

  save_vtk_ascii = false;
  save_vtr_binary = false;
  save_tecplot_ascii = false;

  res = 1.0;

  tau = 1.0;
  step_max = 1000;
  step = 0;

  vtkPeriod = 10;
  statPeriod = 10;

  result_folder = "./RESULT";

  // Phase separation
  rho0 = 0.10;
  IniPerturbRate = 1e-3;
  G = -1.0;

  // EOS
  carn_star = false;
  T_Tc = 0.7;
  rhol_spinodal = 0.2726;
  rhog_spinodal = 0.0484;

  // Density injection
  inject_density = false;
  inject_type = 0;
  inject_position = 1.;
  rho_inject_period = 100;
  rho_increment = 0.;

  read_densities = false;
  hysteresis = false;
  max_saturation = 0.98;
  min_saturation = 0.04;
  wanted_num_threads = 4;

  erosion = false;
}

//! Reserve memory for the computation on both host and device
void MultiphaseLBM::resize_gpu() {
  size_t nbytes_real = lx * ly * lz * sizeof(real);
  size_t nbytes_bool = lx * ly * lz * sizeof(bool);
  size_t nbytes_size_t = lx * ly * lz * sizeof(size_t);

  // Allocate host memory for multidimensional arrays
  std::cout << "Setting domain size to: " << lx << "x" << ly << "x" << lz
            << std::endl;

  liq_cluster_number = (Size_txyz*)malloc(nbytes_size_t);
  liq_cluster = (Boolxyz*)malloc(nbytes_bool);
  gas_cluster_number = (Size_txyz*)malloc(nbytes_size_t);
  gas_cluster = (Boolxyz*)malloc(nbytes_bool);
  solid_number = (Size_txyz*)malloc(nbytes_size_t);

  // Allocate device memory for multidimensional arrays
  cudaErrorCheck(cudaMallocManaged((void**)&drho, nbytes_real));
  cudaErrorCheck(cudaMallocManaged((void**)&dpressure, nbytes_real));
  cudaErrorCheck(cudaMallocManaged((void**)&dis_fluid, nbytes_bool));

  cudaErrorCheck(cudaMalloc((void**)&dpsi, nbytes_real));
  cudaErrorCheck(cudaMalloc((void**)&dcollision_f, nbytes_real * Q));
  cudaErrorCheck(cudaMalloc((void**)&dstream_f, nbytes_real * Q));
  cudaErrorCheck(cudaMalloc((void**)&dforce, nbytes_real * 3));
  cudaErrorCheck(cudaMallocManaged((void**)&dsolid_touch_liquid, nbytes_bool));

  cudaDeviceSynchronize();

  // Initializing host and device memory
  if (read_densities) {
    std::cout << "Reading density data from file: " << densities_filename
              << std::endl;
    io::LineReader in(densities_filename);
    int k = 0;
    bool line_read = false;
    while (char* line = in.next_line()) {
      std::string str_line(line);
      if (str_line == "SCALARS Pressure float 1") break;
      if (str_line == "SCALARS Density float 1") line_read = true;
      if (line_read) {
        if (str_line != "SCALARS Density float 1" &&
            str_line != "LOOKUP_TABLE default") {
          size_t x = k % lx;
          size_t y = (k / lx) % ly;
          size_t z = k / lx / ly;
          drho[x][y][z] = std::stod(line);
          k++;
        }
      }
    }
  } else {
    srand(1);
    for (size_t z = 0; z < lz; ++z) {
      for (size_t y = 0; y < ly; ++y) {
        for (size_t x = 0; x < lx; ++x) {

          drho[x][y][z] =
              rho0 *
              (1.0 + IniPerturbRate * ((real)rand() / (real)RAND_MAX - 0.5));
        }
      }
    }
  }
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        dis_fluid[x][y][z] = true;
        dsolid_touch_liquid[x][y][z] = false;
        liq_cluster_number[x][y][z] = 0;
        liq_cluster[x][y][z] = false;
        solid_number[x][y][z] = 0;
      }
    }
  }
}

//! \brief Perfrom a depth-first search to find a cluster
//! \param[in] x x-coordinate of the first node of the cluster
//! \param[in] y y-coordinate of the first node of the cluster
//! \param[in] z z-coordinate of the first node of the cluster
void MultiphaseLBM::depth_first_search(size_t x, size_t y, size_t z,
                                       ClusterType type) {
  size_t xn, yn, zn;
  size_t xp, yp, zp;
  if (cluster_remove) {
    if (type == ClusterType::liq) {
      if (liq_cluster_number[x][y][z] != liq_cluster_count) return;
      liq_cluster_number[x][y][z] = 0;
    } else if (type == ClusterType::gas) {
      if (gas_cluster_number[x][y][z] != gas_cluster_count) return;
      gas_cluster_number[x][y][z] = 0;
    }
  } else {
    if (type == ClusterType::liq) {
      if (!liq_cluster[x][y][z]) {
        if (solid_number[x][y][z] != 0) {
          cluster_to_solids_map[liq_cluster_count].insert(
              solid_number[x][y][z]);
        }
        return;
      }
      liq_cluster[x][y][z] =
          false;  // indicating that the node has been visited
      liq_cluster_number[x][y][z] = liq_cluster_count;
      cluster_size += 1;
    } else if (type == ClusterType::gas) {
      if (!gas_cluster[x][y][z]) return;
      gas_cluster[x][y][z] =
          false;  // indicating that the node has been visited
      gas_cluster_number[x][y][z] = gas_cluster_count;
      cluster_size += 1;
    }
  }

  // current implementation for periodic boundary only
  xp = (x > 0) ? (x - 1) : (lx - 1);
  xn = (x < lx - 1) ? (x + 1) : (0);
  yp = (y > 0) ? (y - 1) : (ly - 1);
  yn = (y < ly - 1) ? (y + 1) : (0);
  zp = (z > 0) ? (z - 1) : (lz - 1);
  zn = (z < lz - 1) ? (z + 1) : (0);

  depth_first_search(xp, y, z, type);
  depth_first_search(x, yp, z, type);
  depth_first_search(x, y, zp, type);
  // depth_first_search(xp, yp, z);
  // depth_first_search(xp, yn, z);
  // depth_first_search(xp, y, zp);
  // depth_first_search(xp, y, zn);
  // depth_first_search(x, yp, zp);
  // depth_first_search(x, yp, zn);
  depth_first_search(xn, y, z, type);
  depth_first_search(x, yn, z, type);
  depth_first_search(x, y, zn, type);
  // depth_first_search(xn, yn, z);
  // depth_first_search(xn, yp, z);
  // depth_first_search(xn, y, zn);
  // depth_first_search(xn, y, zp);
  // depth_first_search(x, yn, zn);
  // depth_first_search(x, yn, zp);
}

void MultiphaseLBM::calc_cluster() {

  // measuring the liquid and vapor pressures
  size_t fluid_count = 0;
  size_t liquid_count = 0;
  size_t vapor_count = 0;
  real liquid_pressure = 0.;
  real vapor_pressure = 0.;
  real liquid_density = 0.;
  real vapor_density = 0.;
  real d_loc = 0.;

#ifdef _OPENMP
#pragma omp parallel for private(d_loc) reduction(+:fluid_count,liquid_pressure,vapor_pressure,liquid_count,vapor_count,liquid_density,vapor_density) schedule(runtime)
#endif
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (dis_fluid[x][y][z]) {
          fluid_count += 1;
          liq_cluster[x][y][z] = false;
          liq_cluster_number[x][y][z] = 0;
          gas_cluster[x][y][z] = false;
          gas_cluster_number[x][y][z] = 0;
          d_loc = drho[x][y][z];

          if (d_loc > rhol_spinodal) {
            liquid_pressure += dpressure[x][y][z];
            liquid_density += drho[x][y][z];
            liquid_count += 1;
            liq_cluster[x][y][z] = true;
          } else if (d_loc < rhog_spinodal) {
            vapor_pressure += dpressure[x][y][z];
            vapor_density += drho[x][y][z];
            vapor_count += 1;
            gas_cluster[x][y][z] = true;
          }
        }
      }
    }
  }

  if (liquid_count != 0) {
    average_liquid_pressure = liquid_pressure / liquid_count;
    average_liquid_density = liquid_density / liquid_count;
  }
  if (vapor_count != 0) {
    average_vapor_pressure = vapor_pressure / vapor_count;
    average_vapor_density = vapor_density / vapor_count;
  }
  saturation_ratio = float(liquid_count) / float(fluid_count);

  // finding the number of the liquid-solid and liquid-vapor boundary nodes
  size_t n_liquid_solid_boundary = 0;
  size_t n_liquid_vapor_boundary = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:n_liquid_solid_boundary,n_liquid_vapor_boundary) schedule(runtime)
#endif
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (liq_cluster[x][y][z]) {
          for (size_t i = 1; i < Q; ++i) {
            size_t next_x = x - ex[i];
            if (x == 0 && ex[i] == 1) next_x = lx - 1;
            if (x == lx - 1 && ex[i] == -1) next_x = 0;

            size_t next_y = y - ey[i];
            if (y == 0 && ey[i] == 1) next_y = ly - 1;
            if (y == ly - 1 && ey[i] == -1) next_y = 0;

            size_t next_z = z - ez[i];
            if (z == 0 && ez[i] == 1) next_z = lz - 1;
            if (z == lz - 1 && ez[i] == -1) next_z = 0;
            if (!dis_fluid[next_x][next_y][next_z]) {
              n_liquid_solid_boundary += 1;
              break;
            } else if (drho[next_x][next_y][next_z] < rhol_spinodal) {
              n_liquid_vapor_boundary += 1;
              break;
            }
          }
        }
      }
    }
  }
  if (liquid_count > 0) {
    ratio_liquid_solid_boundary =
        (double)n_liquid_solid_boundary / liquid_count;
    ratio_liquid_vapor_boundary =
        (double)n_liquid_vapor_boundary / liquid_count;
  }

  // finding the number of liquid clusters
  liq_cluster_count = 0;
  cluster_to_solids_map.clear();
  size_t min_cluster_size = 7;
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (liq_cluster[x][y][z]) {
          liq_cluster_count += 1;
          depth_first_search(x, y, z, ClusterType::liq);
          if (cluster_size < min_cluster_size) {
            cluster_to_solids_map.erase(liq_cluster_count);
            cluster_remove = true;
            depth_first_search(x, y, z, ClusterType::liq);
            liq_cluster_count -= 1;
            cluster_remove = false;
          }
          cluster_size = 0;
        }
      }
    }
  }

  // finding the number of gas clusters
  gas_cluster_count = 0;
  cluster_size = 0;
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (gas_cluster[x][y][z]) {
          gas_cluster_count += 1;
          depth_first_search(x, y, z, ClusterType::gas);
          if (cluster_size < min_cluster_size) {
            cluster_remove = true;
            depth_first_search(x, y, z, ClusterType::gas);
            gas_cluster_count -= 1;
            cluster_remove = false;
          }
          cluster_size = 0;
        }
      }
    }
  }

  // fining the location and size of the liquid cluster, if there is only one
  if (liq_cluster_count == 1) {
    bool x_cluster_break = false;
    bool y_cluster_break = false;
    bool z_cluster_break = false;
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (liq_cluster_number[0][y][z] == 1) {
          if (liq_cluster_number[lx - 1][y][z] == 1) {
            x_cluster_break = true;
            break;
          }
        }
      }
      if (x_cluster_break == true) break;
    }
    for (size_t x = 0; x < lx; ++x) {
      for (size_t z = 0; z < lz; ++z) {
        if (liq_cluster_number[x][0][z] == 1) {
          if (liq_cluster_number[x][ly - 1][z] == 1) {
            y_cluster_break = true;
            break;
          }
        }
      }
      if (y_cluster_break == true) break;
    }
    for (size_t x = 0; x < lx; ++x) {
      for (size_t y = 0; y < ly; ++y) {
        if (liq_cluster_number[x][y][0] == 1) {
          if (liq_cluster_number[x][y][lz - 1] == 1) {
            z_cluster_break = true;
            break;
          }
        }
      }
      if (z_cluster_break == true) break;
    }
    if (step % vtkPeriod == 0)
      std::cout << x_cluster_break << "\t" << y_cluster_break << "\t"
                << z_cluster_break << std::endl;

    bool vapor_plane;
    size_t x_vapor_plane;
    size_t y_vapor_plane;
    size_t z_vapor_plane;

    if (x_cluster_break) {
      for (size_t x = 0; x < lx; ++x) {
        vapor_plane = true;
        for (size_t y = 0; y < ly; ++y) {
          for (size_t z = 0; z < lz; ++z) {
            if (liq_cluster_number[x][y][z] == 1) {
              vapor_plane = false;
              break;
            }
          }
          if (!vapor_plane) break;
        }
        if (vapor_plane) {
          x_vapor_plane = x;
          break;
        }
      }
    }

    if (y_cluster_break) {
      for (size_t y = 0; y < ly; ++y) {
        vapor_plane = true;
        for (size_t z = 0; z < lz; ++z) {
          for (size_t x = 0; x < lx; ++x) {
            if (liq_cluster_number[x][y][z] == 1) {
              vapor_plane = false;
              break;
            }
          }
          if (!vapor_plane) break;
        }
        if (vapor_plane) {
          y_vapor_plane = y;
          break;
        }
      }
    }

    if (z_cluster_break) {
      for (size_t z = 0; z < lz; ++z) {
        vapor_plane = true;
        for (size_t x = 0; x < lx; ++x) {
          for (size_t y = 0; y < ly; ++y) {
            if (liq_cluster_number[x][y][z] == 1) {
              vapor_plane = false;
              break;
            }
          }
          if (!vapor_plane) break;
        }
        if (vapor_plane) {
          z_vapor_plane = z;
          break;
        }
      }
    }

    size_t adjusted_x;
    size_t adjusted_y;
    size_t adjusted_z;
    size_t sumx = 0;
    size_t sumy = 0;
    size_t sumz = 0;
    size_t size = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sumx,sumy,sumz,size) private(adjusted_x,adjusted_y,adjusted_z) schedule(runtime)
#endif
    for (size_t x = 0; x < lx; ++x) {
      for (size_t y = 0; y < ly; ++y) {
        for (size_t z = 0; z < lz; ++z) {
          if (liq_cluster_number[x][y][z] == 1) {
            adjusted_x = x;
            adjusted_y = y;
            adjusted_z = z;
            if (x_cluster_break && x < x_vapor_plane) adjusted_x += lx;
            if (y_cluster_break && y < y_vapor_plane) adjusted_y += ly;
            if (z_cluster_break && z < z_vapor_plane) adjusted_z += lz;
            // std::cout<<adjusted_x<<"\t"<<adjusted_y<<"\t"<<adjusted_z<<"\t"<<std::endl;
            sumx += adjusted_x;
            sumy += adjusted_y;
            sumz += adjusted_z;
            size += 1;
          }
        }
      }
    }

    real x_center = sumx / size;
    real y_center = sumy / size;
    real z_center = sumz / size;
    if (step % vtkPeriod == 0)
      std::cout << x_center << "\t" << y_center << "\t" << z_center
                << std::endl;

    max_r = 0;
    real r2;
#ifdef _OPENMP
#pragma omp parallel for reduction( \
    max                             \
    : max_r) private(r2, adjusted_x, adjusted_y, adjusted_z) schedule(runtime)
#endif
    for (size_t x = 0; x < lx; ++x) {
      for (size_t y = 0; y < ly; ++y) {
        for (size_t z = 0; z < lz; ++z) {
          if (liq_cluster_number[x][y][z] == 1) {
            adjusted_x = x;
            adjusted_y = y;
            adjusted_z = z;
            if (x_cluster_break && x < x_vapor_plane) adjusted_x += lx;
            if (y_cluster_break && y < y_vapor_plane) adjusted_y += ly;
            if (z_cluster_break && z < z_vapor_plane) adjusted_z += lz;
            // std::cout<<adjusted_x<<"\t"<<adjusted_y<<"\t"<<adjusted_z<<"\t"<<std::endl;
            r2 = (x_center - adjusted_x) * (x_center - adjusted_x) +
                 (y_center - adjusted_y) * (y_center - adjusted_y) +
                 (z_center - adjusted_z) * (z_center - adjusted_z);
            if (sqrt(r2) > max_r) {
              max_r = sqrt(r2);
            }
          }
        }
      }
    }
    if (step % vtkPeriod == 0) std::cout << "radius: " << max_r << std::endl;
  } else
    max_r = 0;

  // calculating the cluster order (m)
  cluster_order = 0;
  for (std::map<size_t, std::set<size_t>>::iterator map_it =
           cluster_to_solids_map.begin();
       map_it != cluster_to_solids_map.end(); ++map_it)
    cluster_order += map_it->second.size();
  if (liq_cluster_count != 0) cluster_order /= cluster_to_solids_map.size();

  // creating map of solid number to cluster numbers
  solid_to_clusters_map.clear();
  for (std::map<size_t, std::set<size_t>>::iterator map_it =
           cluster_to_solids_map.begin();
       map_it != cluster_to_solids_map.end(); ++map_it) {
    for (std::set<size_t>::iterator solid_set_it = map_it->second.begin();
         solid_set_it != map_it->second.end(); ++solid_set_it) {
      solid_to_clusters_map[*solid_set_it].insert(map_it->first);
    }
  }

  // calculating the wet coordination number (n)
  wet_coord_num = 0;
  for (std::map<size_t, std::set<size_t>>::iterator map_it =
           solid_to_clusters_map.begin();
       map_it != solid_to_clusters_map.end(); ++map_it) {
    wet_coord_num += map_it->second.size();
  }
  if (solid_count != 0) wet_coord_num /= solid_count;
}

//! Run the program on gpu
void MultiphaseLBM::run_gpu() {
  size_t numFile_vtk = 0;
  // size_t numFile_vtr = 0;
  // size_t numFile_tecplot = 0;
  // size_t numZpressFile = 0;
  // control the direction of injection
  // reverse = 1: do not reverse
  // reverse = -1: reverse
  real reverse = 1;

  size_t nbytes_real = lx * ly * lz * sizeof(real);
  size_t nbytes_bool = lx * ly * lz * sizeof(bool);

  // Prefetch the data to the GPU
  int device = 1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(drho, nbytes_real, device, 0);
  cudaMemPrefetchAsync(dis_fluid, nbytes_bool, device, 0);

  transfer_parameter(&tau, &G, &carn_star, &T_Tc, &rhol_spinodal,
                     &rhog_spinodal, &inject_density, &rho_inject_period,
                     &rho_increment, &rhos, &inject_position, &statPeriod);

  multiphaselbm_initialization(&drho, &dpsi, &dpressure, &dis_fluid,
                               &dcollision_f, &dstream_f, &dforce);
  cudaDeviceSynchronize();
  double td = get_time();
  for (; step < step_max; step++) {
    multiphaselbm_iteration(step, statPeriod, reverse, inject_type);

    if ((step) % (statPeriod) == 0) {

      calc_cluster();
      printf("saturation=%f, aLp=%f,aVp=%f, step=%d\n", saturation_ratio,
             average_liquid_pressure, average_vapor_pressure, step);
      write_stats();
      if (save_vtk_ascii && step % vtkPeriod == 0) {
        std::cout << "Saving vtk ascii file number " << numFile_vtk
                  << std::endl;
        write_vtk_ascii(numFile_vtk);
        numFile_vtk++;
      }
    }
  };

  td = get_time() - td;
  printf("GPU time for %d steps on resolution of %d: %f sec\n", step_max, lx,
         td);
  std::cout << std::endl;
}

//! Read an input file with the physical parameters and
// the description of the system. See MultiphaseLBM documentation for detailed
// description of the different tokens.
void MultiphaseLBM::read_data(const char* name) {
  std::ifstream file(name);
  if (file.fail()) {
    std::cerr << "Cannot open command file " << name << std::endl;
  }

  std::string token;
  file >> token;

  char sample_filename[256];
  char positions_filename[256];

  while (file.good()) {
    if (token[0] == '!' || token[0] == '#')
      getline(file, token);
    else if (token == "result_folder")
      file >> result_folder;
    else if (token == "num_threads")
      file >> wanted_num_threads;
    else if (token == "step")
      file >> step;
    else if (token == "step_max")
      file >> step_max;
    else if (token == "vtkPeriod") {
      file >> vtkPeriod;
      save_vtk_ascii = true;
    } else if (token == "vtrPeriod") {
      file >> vtrPeriod;
      save_vtr_binary = true;
    } else if (token == "tecplotPeriod") {
      file >> tecplotPeriod;
      save_tecplot_ascii = true;
    } else if (token == "statPeriod")
      file >> statPeriod;
    else if (token == "rho0")
      file >> rho0;
    else if (token == "rhos")
      file >> rhos;
    else if (token == "G")
      file >> G;
    else if (token == "IniPerturbRate")
      file >> IniPerturbRate;
    else if (token == "tau")
      file >> tau;
    else if (token == "resize")
      resize_gpu();
    else if (token == "read_sample") {
      file >> sample_filename;
      std::cout << "Reading node ids from: " << sample_filename << std::endl;
      read_sample(sample_filename);
    } else if (token == "place_sphere") {
      real x, y, z, R;
      file >> x >> y >> z >> R;
      place_sphere(x, y, z, R);
      std::cout << "Placing sphere with radius " << R << " at (" << x << ", "
                << y << ", " << z << ")" << std::endl;
    } else if (token == "generate_solids") {
      int num;
      real rmin, rmax, xmin, xmax, ymin, ymax, zmin, zmax;
      file >> num >> rmin >> rmax >> xmin >> xmax >> ymin >> ymax >> zmin >>
          zmax;
      generate_solids(num, rmin, rmax, xmin, xmax, ymin, ymax, zmin, zmax);
      std::cout << "Generating grains with radii between " << rmin << " and "
                << rmax << std::endl;
    } else if (token == "resolution")
      file >> res;
    else if (token == "read_positions") {
      file >> positions_filename;
      std::cout << "Reading grain positions and radii from: "
                << positions_filename << "\nwith resolution of: " << res
                << std::endl;
      read_positions(positions_filename);
      calc_packing_density();
    } else if (token == "rho_inject_period") {
      file >> rho_inject_period;
    } else if (token == "rho_increment") {
      inject_density = true;
      file >> rho_increment;
    } else if (token == "inject_type") {
      file >> inject_type;
    } else if (token == "inject_position") {
      file >> inject_position;
    } else if (token == "carn_star") {
      carn_star = true;
    } else if (token == "T_Tc") {
      file >> T_Tc;
    } else if (token == "rhol_spinodal") {
      file >> rhol_spinodal;
    } else if (token == "rhog_spinodal") {
      file >> rhog_spinodal;
    } else if (token == "read_densities") {
      file >> densities_filename;
      read_densities = true;
    } else if (token == "max_saturation") {
      file >> max_saturation;
      hysteresis = true;
    } else if (token == "min_saturation") {
      file >> min_saturation;
    } else if (token == "target_density") {
      file >> target_density;
      erosion = true;
    } else {
      std::cerr << "The token " << token << "is undefined." << std::endl;
    }

    file >> token;
  }
  std::cout << "rho0: " << rho0 << std::endl;
  std::cout << "perturbation rate: " << IniPerturbRate << std::endl;
  std::cout << "rhos: " << rhos << std::endl;
  if (carn_star)
    std::cout << "Using Carnahan-Starling EOS with T/Tc = " << T_Tc
              << ", rhol* = " << rhol_spinodal
              << " and rhog* = " << rhog_spinodal << std::endl;
  else
    std::cout << "Using Shan-Chen EOS with G = " << G << std::endl;
  std::cout << "Injecting density with an increment of " << rho_increment
            << " every " << rho_inject_period << " step." << std::endl;
  std::cout << "Injection type: " << inject_type << std::endl;

  find_solid_touch_liquid();
  if (erosion) erode_solid_touch_liquid();
#ifdef _OPENMP
  omp_set_num_threads(wanted_num_threads);
  std::cout << "Number of OpenMP threads = " << wanted_num_threads << std::endl;
#endif
}

//! Read a sample as a 3D-matrix of integers, 1 for solid 0 for liquid.
// First three lines should be the size of the grid in the three directions.
//! \param[in] name Name of the file
void MultiphaseLBM::read_sample(const char* name) {
  std::ifstream file(name);
  if (file.fail()) {
    std::cerr << "Cannot open sample file " << name << std::endl;
  }

  std::vector<bool> ids;
  std::cout << "Reading sample file: " << name << std::endl;

  int lxfile;
  int lyfile;
  int lzfile;

  if (file.is_open() && file.good()) {
    std::string line;
    file >> lxfile;
    file >> lyfile;
    file >> lzfile;
    unsigned down_sample = lxfile / lx;
    resize_gpu();
    ids.reserve(1000);

    // Read the sample
    unsigned k = 0;
    int x, y, z;
    while (std::getline(file, line)) {
      std::istringstream istream(line);
      int vid;
      x = k / lzfile / lyfile;
      y = (k / lzfile) % lyfile;
      z = k % lzfile;
      if (x % down_sample == 0 && y % down_sample == 0 &&
          z % down_sample == 0) {
        istream >> vid;
        ids.emplace_back(vid);
      }
      ++k;
    }
  }
  std::cout << __FILE__ << __LINE__ << std::endl;
  file.close();
  std::cout << __FILE__ << __LINE__ << std::endl;
  bool ID;
  unsigned k = 0;
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        ID = ids.at(k);
        dis_fluid[x][y][z] = !ID;  // ID of 1 is solid and 0 is fluid
        ++k;
      }
    }
  }
  std::cout << __FILE__ << __LINE__ << std::endl;

  calc_packing_density();
}

//! Define the fluid-solid boundary inside the sample and at the domain
// boundaries
void MultiphaseLBM::find_solid_touch_liquid() {
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        dsolid_touch_liquid[x][y][z] = false;

        if (!dis_fluid[x][y][z]) {
          for (size_t i = 1; i < Q; ++i) {
            size_t next_x = x - ex[i];
            if (x == 0 && ex[i] == 1) next_x = lx - 1;
            if (x == lx - 1 && ex[i] == -1) next_x = 0;

            size_t next_y = y - ey[i];
            if (y == 0 && ey[i] == 1) next_y = ly - 1;
            if (y == ly - 1 && ey[i] == -1) next_y = 0;

            size_t next_z = z - ez[i];
            if (z == 0 && ez[i] == 1) next_z = lz - 1;
            if (z == lz - 1 && ez[i] == -1) next_z = 0;

            if (dis_fluid[next_x][next_y][next_z]) {
              dsolid_touch_liquid[x][y][z] = true;
            }
          }
        }
      }
    }
  }
}

//! Define the fluid-solid boundary inside the sample and at the domain
// boundaries
void MultiphaseLBM::erode_solid_touch_liquid() {

  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        if (dsolid_touch_liquid[x][y][z]) solid_liquid_num++;
      }
    }
  }

  size_t eroded_node_num = (1 - target_density / packing_density) * solids;
  std::cout << "Eroded solids number: " << eroded_node_num << std::endl;
  std::cout << "Solid touch liquid number: " << solid_liquid_num << std::endl;

  while (eroded_node_num > solid_liquid_num) {
    for (size_t z = 0; z < lz; ++z) {
      for (size_t y = 0; y < ly; ++y) {
        for (size_t x = 0; x < lx; ++x) {
          if (dsolid_touch_liquid[x][y][z]) {
            dsolid_touch_liquid[x][y][z] = false;
            dis_fluid[x][y][z] = true;
            eroded_node_num--;
            solid_liquid_num--;
          }
        }
      }
    }
    find_solid_touch_liquid();
  }

  if (eroded_node_num <= solid_liquid_num && eroded_node_num > 0) {
    real erosion_rate = double(eroded_node_num) / double(solid_liquid_num);
    for (size_t z = 0; z < lz; ++z) {
      for (size_t y = 0; y < ly; ++y) {
        for (size_t x = 0; x < lx; ++x) {
          if (dsolid_touch_liquid[x][y][z] &&
              (real)rand() / (real)RAND_MAX < erosion_rate) {
            dsolid_touch_liquid[x][y][z] = false;
            dis_fluid[x][y][z] = true;
            --eroded_node_num;
            solid_liquid_num--;
          }
        }
      }
    }
  }
  find_solid_touch_liquid();
  calc_packing_density();
}

//! Place a single solid sphere in the domain
void MultiphaseLBM::place_sphere(real x, real y, real z, real R) {
  real xmin, xmax;
  real ymin, ymax;
  real zmin, zmax;

  xmin = x - R;
  ymin = y - R;
  zmin = z - R;

  xmax = x + R;
  ymax = y + R;
  zmax = z + R;

  for (real px = xmin; px <= xmax; px += 1.) {
    for (real py = ymin; py <= ymax; py += 1.0) {
      for (real pz = zmin; pz <= zmax; pz += 1.0) {
        real dx = px - x;
        real dy = py - y;
        real dz = pz - z;
        real dist2 = dx * dx + dy * dy + dz * dz;
        real R2 = R * R;
        if (dist2 < R2) {
          int near_px = (nearest(px) < 0) ? nearest(px) + lx : nearest(px);
          int near_py = (nearest(py) < 0) ? nearest(py) + ly : nearest(py);
          int near_pz = (nearest(pz) < 0) ? nearest(pz) + lz : nearest(pz);
          if (near_px >= lx) near_px -= lx;
          if (near_py >= ly) near_py -= ly;
          if (near_pz >= lz) near_pz -= lz;
          dis_fluid[near_px][near_py][near_pz] = false;
          solid_number[near_px][near_py][near_pz] = solid_count;
        }
      }
    }
  }
}

//! Initialize MultiphaseLBM
void MultiphaseLBM::init() {
  create_folder(result_folder);
  std::cout << "Resulting files will be saved in " << result_folder
            << std::endl;
  char name[256];
  sprintf(name, "%s/Stats.txt", result_folder.c_str());
  statFile.open(name);
  statFile
      << "step\tmin_rho\tmax_rho\tmin_p\tmax_p\t"
         "avg_liq_press\tavg_vap_press\t"
         "suction\tSr\tavg_liq_rho\tavg_vap_rho\t"
         "liq_cluster_count\tcluster_order\twet_coord_num\tgas_cluster_"
         "count\tliquid_solid_boundary\tliquid_vapor_boundary\tcluster_radius"
      << std::endl;
}

//! Output the fluid flow and solid domain over time in
// resultXXXXXX.vtk file.
void MultiphaseLBM::write_vtk_ascii(size_t num) {
  real pasxyz;  //, u_x, u_y, u_z;
  FILE* sortie;
  char nomfic[256];
  sprintf(nomfic, "%s/result%06zu.vtk", result_folder.c_str(), num);
  pasxyz = 1.0 / lx;
  sortie = fopen(nomfic, "w");
  fprintf(sortie, "# vtk DataFile Version 2.0\n");
  fprintf(sortie, "Sortie domaine LB+LINK\n");
  fprintf(sortie, "ASCII\n");
  fprintf(sortie, "DATASET RECTILINEAR_GRID\n");
  fprintf(sortie, "DIMENSIONS %zu %zu %zu\n", lx, ly, lz);

  fprintf(sortie, "X_COORDINATES %zu float\n", lx);
  for (size_t i = 0; i <= lx - 1; ++i) {
    fprintf(sortie, "%.4e ", (real)i * pasxyz);
  }
  fprintf(sortie, "\n");

  fprintf(sortie, "Y_COORDINATES %zu float\n", ly);
  for (size_t i = 0; i <= ly - 1; ++i) {
    fprintf(sortie, "%.4e ", (real)i * pasxyz);
  }
  fprintf(sortie, "\n");

  fprintf(sortie, "Z_COORDINATES %zu float\n", lz);
  for (size_t i = 0; i <= lz - 1; ++i) {
    fprintf(sortie, "%e ", (float)i * pasxyz);
  }
  fprintf(sortie, "\n");

  fprintf(sortie, "POINT_DATA %zu\n", lx * ly * lz);

  fprintf(sortie, "SCALARS Zones int 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  int id;

  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        if (dis_fluid[x][y][z])
          id = 0;
        else
          id = 1;
        fprintf(sortie, "%d\n", id);
      }
    }
  }

  fprintf(sortie, "SCALARS Density float 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        if (dis_fluid[x][y][z])
          fprintf(sortie, "%.4e\n", (drho[x][y][z]));
        else
          fprintf(sortie, "0\n");
      }
    }
  }

  fprintf(sortie, "SCALARS Pressure float 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        if (dis_fluid[x][y][z]) {
          fprintf(sortie, "%.4e\n", (dpressure[x][y][z]));
        } else
          fprintf(sortie, "0\n");
      }
    }
  }

  fprintf(sortie, "SCALARS LiqClusters int 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        fprintf(sortie, "%ld\n", liq_cluster_number[x][y][z]);
      }
    }
  }

  fprintf(sortie, "SCALARS GasClusters int 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        fprintf(sortie, "%ld\n", gas_cluster_number[x][y][z]);
      }
    }
  }

  fprintf(sortie, "SCALARS Solids int 1\n");
  fprintf(sortie, "LOOKUP_TABLE default\n");
  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        fprintf(sortie, "%ld\n", solid_number[x][y][z]);
      }
    }
  }

  fclose(sortie);
}

//! Write model statistics to the Stats.txt file
void MultiphaseLBM::write_stats() {
  real max_density = -1e-9, min_density = 1e9;
  real max_pressure = -1e-9, min_pressure = 1e9;

  for (size_t z = 0; z < lz; ++z) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t x = 0; x < lx; ++x) {
        if (dis_fluid[x][y][z]) {
          min_density = std::min(drho[x][y][z], min_density);
          max_density = std::max(drho[x][y][z], max_density);
          min_pressure = std::min(dpressure[x][y][z], min_pressure);
          max_pressure = std::max(dpressure[x][y][z], max_pressure);
        }
      }
    }
  }
  statFile << step << "\t" << min_density << "\t" << max_density << "\t"
           << min_pressure << "\t" << max_pressure << "\t"
           << average_liquid_pressure << "\t" << average_vapor_pressure << "\t"
           << average_vapor_pressure - average_liquid_pressure << "\t"
           << saturation_ratio << "\t" << average_liquid_density << "\t"
           << average_vapor_density << "\t" << liq_cluster_count << "\t"
           << cluster_order << "\t" << wet_coord_num << "\t"
           << gas_cluster_count << "\t" << ratio_liquid_solid_boundary << "\t"
           << ratio_liquid_vapor_boundary << "\t" << max_r << std::endl
           << std::flush;
}

//! Generate uniformly distirbured particles
void MultiphaseLBM::generate_solids(int num, real rmin, real rmax, real xmin,
                                    real xmax, real ymin, real ymax, real zmin,
                                    real zmax) {
  std::cout << "generating " << num
            << " uniformly distributed particles with radii between " << rmin
            << " and " << rmax << " in a box with dimension (" << xmin << ","
            << xmax << ") (" << ymin << "," << ymax << ") (" << zmin << ","
            << zmax << ")" << std::endl;
  real r, x, y, z;
  for (int i = 0; i < num; ++i) {
    r = (rmax - rmin) * ((real)rand() / (real)RAND_MAX) + rmin;
    x = (xmax - xmin) * ((real)rand() / (real)RAND_MAX) + xmin;
    y = (ymax - ymin) * ((real)rand() / (real)RAND_MAX) + ymin;
    z = (zmax - zmin) * ((real)rand() / (real)RAND_MAX) + zmin;
    solid_count += 1;
    place_sphere(x, y, z, r);
  }
}

//! Read position of spherical particles from file
void MultiphaseLBM::read_positions(const char* name) {
  real x, y, z, r;
  int i = 0;
  io::LineReader in(name);
  while (char* line = in.next_line()) {
    ++i;
    if (i == 1)
      x = std::stod(line) * res;
    else if (i == 2)
      y = std::stod(line) * res;
    else if (i == 3)
      z = std::stod(line) * res;
    else {
      r = std::stod(line) * res;
      solid_count += 1;
      place_sphere(x, y, z, r);
      i = 0;
    }
  }
  std::cout << "number of grain = " << solid_count << std::endl;
}

//! Calculate the packing density of the specimen
void MultiphaseLBM::calc_packing_density() {
  solids = 0;
  for (size_t x = 0; x < lx; ++x) {
    for (size_t y = 0; y < ly; ++y) {
      for (size_t z = 0; z < lz; ++z) {
        if (!dis_fluid[x][y][z]) {
          solids++;
        }
      }
    }
  }
  packing_density = float(solids) / (lx * ly * lz);
  std::cout << "Packing density: " << packing_density << std::endl;
}

//! Release memories of both host and device multidimensional arrays
MultiphaseLBM::~MultiphaseLBM() {

  cudaFree(drho);
  cudaFree(dpsi);
  cudaFree(dpressure);
  cudaFree(dcollision_f);
  cudaFree(dstream_f);
  cudaFree(dforce);
  cudaFree(dis_fluid);

  free(liq_cluster);
  free(gas_cluster);
  free(liq_cluster_number);
  free(gas_cluster_number);
  free(solid_number);
}
