#ifndef ITERATION_KERNEL_CUH_
#define ITERATION_KERNEL_CUH_
#include "iteration_device.cuh"

//! Initialize multidimensional arrays on device
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
__global__ void multiphaselbm_init() {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < lx && y < ly && z < lz) {
    for (size_t i = 0; i < Q; i++) {
      collision_f_d[x][y][z][i] = t[i] * rho_d[x][y][z];
      stream_f_d[x][y][z][i] = t[i] * rho_d[x][y][z];
    }
    if (is_fluid_d[x][y][z]) {
      psi_d[x][y][z] = dPsi(rho_d[x][y][z]);
      pressure_d[x][y][z] = dPress(rho_d[x][y][z]);
    } else {
      rho_d[x][y][z] = drhos;
      psi_d[x][y][z] = dPsi(drhos);
      force_d[x][y][z][0] = 0.0;
      force_d[x][y][z][1] = 0.0;
      force_d[x][y][z][2] = 0.0;
    }
  }
}

//! Calculating rho, psi and pressure based on the distributions
__global__ void do_collision(long step, int reverse) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (is_fluid_d[x][y][z] && x < lx && y < ly && z < lz) {
    compute_density(step, x, y, z, reverse);
    compute_force(x, y, z);
    compute_velocity(x, y, z);
  }
}

//! Copy the stream pdfs with the collision pdfs for next iteration
__global__ void do_swap() {
  RealxyzQ* swap = collision_f_d;
  collision_f_d = stream_f_d;
  stream_f_d = swap;
}

//! Bounce back in obstacles
__global__ void do_streaming() {
  size_t x_Q = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;

  size_t i_min = 1 + int(x_Q / lx) * (Q - 1) / SCALE_LX;
  size_t i_max = i_min + (Q - 1) / SCALE_LX;
  size_t x = x_Q % lx;

  if (is_fluid_d[x][y][z] && x_Q < (SCALE_LX * lx) && y < ly && z < lz)
    stream_from_neighbor(x, y, z, i_min, i_max);
}

//! Apply specified boundary conditions at outer surface
__global__ void treat_boundary() {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  size_t next_x, next_y, next_z;
  if (is_fluid_d[x][y][z] && x < lx && y < ly && z < lz) {
    stream_f_d[x][y][z][0] = collision_f_d[x][y][z][0];
    for (size_t i = 1; i < Q; i++) {
      next_x = x - ex[i];
      if (x == 0 && ex[i] == 1) {
        next_x = lx - 1;
        outer_boundary(x, y, z, i, xmin_bounce_back, xmin_pressure_condition,
                       xmin_open_condition, periodic_x);
      }

      if (x == lx - 1 && ex[i] == -1) {
        next_x = 0;
        outer_boundary(x, y, z, i, xmax_bounce_back, xmax_pressure_condition,
                       xmax_open_condition, periodic_x);
      }

      next_y = y - ey[i];
      if (y == 0 && ey[i] == 1) {
        next_y = ly - 1;
        outer_boundary(x, y, z, i, ymin_bounce_back, ymin_pressure_condition,
                       ymin_open_condition, periodic_y);
      }

      if (y == ly - 1 && ey[i] == -1) {
        next_y = 0;
        outer_boundary(x, y, z, i, ymax_bounce_back, ymax_pressure_condition,
                       ymax_open_condition, periodic_y);
      }

      next_z = z - ez[i];
      if (z == 0 && ez[i] == 1) {
        next_z = lz - 1;
        outer_boundary(x, y, z, i, zmin_bounce_back, zmin_pressure_condition,
                       zmin_open_condition, periodic_z);
      }

      if (z == lz - 1 && ez[i] == -1) {
        next_z = 0;
        outer_boundary(x, y, z, i, zmax_bounce_back, zmax_pressure_condition,
                       zmax_open_condition, periodic_z);
      }

      if (!is_fluid_d[next_x][next_y][next_z]) {
        size_t switched_i = i > half ? (i - half) : (i + half);
        stream_f_d[x][y][z][i] = collision_f_d[x][y][z][switched_i];
      }
    }
  }
}

#endif  // ITERATION_KERNEL_CUH_
