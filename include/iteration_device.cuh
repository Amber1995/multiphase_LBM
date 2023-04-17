#ifndef ITERATION_DEVICE_CUH_
#define ITERATION_DEVICE_CUH_
#include "global_var.cuh"

//! Calculate the pressure for a given density using the chosen EOS
//! \param[in] rhoxyz density
__device__ real dPress(real rhoxyz) {
  if (dcarn_star) {  // Carnahan-Starling EOS
    real a = 1.0;
    real b = 4.0;
    real R = 1.0;
    real Tc = 0.0943;
    real T = dT_Tc * Tc;
    real eta = b * rhoxyz / 4.0;
    real eta2 = eta * eta;
    real eta3 = eta2 * eta;
    real rho2 = rhoxyz * rhoxyz;
    real one_minus_eta = (1.0 - eta);
    real one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta;
    return rhoxyz * R * T * (1 + eta + eta2 - eta3) / one_minus_eta3 - a * rho2;
  } else {  // Shan-Chen EOS
    real cs2 = 1.0 / 3.0;
    real psi = 1. - exp(-rhoxyz);
    real psi2 = psi * psi;
    return cs2 * rhoxyz + cs2 * dG / 2 * psi2;
  }
}

//! Calculate the pressure for a given density using the chosen EOS
//! \param[in] rhoxyz density
__device__ real dPsi(real rhoxyz) {
  if (dcarn_star) {
    real cs2 = 1.0 / 3.0;
    real p = dPress(rhoxyz);
    return sqrt(2.0 * (p - cs2 * rhoxyz) / (cs2 * dG));
  } else
    return 1. - exp(-rhoxyz);
}

__device__ void compute_density(long step, size_t x, size_t y, size_t z,
                                int reverse) {
  // Calculating rho, psi and pressure based on the distributions
  real d_loc = 0.0;
  for (size_t i = 0; i < Q; i++) d_loc += stream_f_d[x][y][z][i];

  // density injection
  if (y < dinject_position * ly && step % drho_inject_period == 0) {
    switch (dinject_type) {
      case 0:
        d_loc += drho_increment * reverse;
        break;
      case 1:
        if (d_loc < drhol_spinodal) d_loc += drho_increment * reverse;
        break;
      case 2:
        if (d_loc > drhol_spinodal) d_loc += drho_increment * reverse;
        break;
    }
  }
  rho_d[x][y][z] = d_loc;
  pressure_d[x][y][z] = dPress(d_loc);
  psi_d[x][y][z] = dPsi(rho_d[x][y][z]);
}

//! Compute force at current node
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
__device__ void compute_force(size_t x, size_t y, size_t z) {
  size_t xn, yn, zn;
  size_t xp, yp, zp;
  unsigned x_stride = blockDim.x * gridDim.x;
  unsigned y_stride = blockDim.y * gridDim.y;
  unsigned z_stride = blockDim.z * gridDim.z;

  for (; x < lx; x += x_stride) {
    xp = (x > 0) ? (x - 1) : (lx - 1);
    xn = (x < lx - 1) ? (x + 1) : (0);
    if ((xp == lx - 1 || xn == 0) && !periodic_x) continue;
    for (; y < ly; y += y_stride) {
      yp = (y > 0) ? (y - 1) : (ly - 1);
      yn = (y < ly - 1) ? (y + 1) : (0);
      if ((yp == ly - 1 || yn == 0) && !periodic_y) continue;
      for (; z < lz; z += z_stride) {
        zp = (z > 0) ? (z - 1) : (lz - 1);
        zn = (z < lz - 1) ? (z + 1) : (0);
        if ((zp == lz - 1 || zn == 0) && !periodic_z) continue;
        if (is_fluid_d[x][y][z]) {
          force_d[x][y][z][0] =
              -dG * psi_d[x][y][z] *
              (w[1] * psi_d[xp][y][z] * ex[1] + w[2] * psi_d[x][yp][z] * ex[2] +
               w[3] * psi_d[x][y][zp] * ex[3] +
               w[4] * psi_d[xp][yp][z] * ex[4] +
               w[5] * psi_d[xp][yn][z] * ex[5] +
               w[6] * psi_d[xp][y][zp] * ex[6] +
               w[7] * psi_d[xp][y][zn] * ex[7] +
               w[8] * psi_d[x][yp][zp] * ex[8] +
               w[9] * psi_d[x][yp][zn] * ex[9] +
               w[10] * psi_d[xn][y][z] * ex[10] +
               w[11] * psi_d[x][yn][z] * ex[11] +
               w[12] * psi_d[x][y][zn] * ex[12] +
               w[13] * psi_d[xn][yn][z] * ex[13] +
               w[14] * psi_d[xn][yp][z] * ex[14] +
               w[15] * psi_d[xn][y][zn] * ex[15] +
               w[16] * psi_d[xn][y][zp] * ex[16] +
               w[17] * psi_d[x][yn][zn] * ex[17] +
               w[18] * psi_d[x][yn][zp] * ex[18]);

          force_d[x][y][z][1] =
              -dG * psi_d[x][y][z] *
              (w[1] * psi_d[xp][y][z] * ey[1] + w[2] * psi_d[x][yp][z] * ey[2] +
               w[3] * psi_d[x][y][zp] * ey[3] +
               w[4] * psi_d[xp][yp][z] * ey[4] +
               w[5] * psi_d[xp][yn][z] * ey[5] +
               w[6] * psi_d[xp][y][zp] * ey[6] +
               w[7] * psi_d[xp][y][zn] * ey[7] +
               w[8] * psi_d[x][yp][zp] * ey[8] +
               w[9] * psi_d[x][yp][zn] * ey[9] +
               w[10] * psi_d[xn][y][z] * ey[10] +
               w[11] * psi_d[x][yn][z] * ey[11] +
               w[12] * psi_d[x][y][zn] * ey[12] +
               w[13] * psi_d[xn][yn][z] * ey[13] +
               w[14] * psi_d[xn][yp][z] * ey[14] +
               w[15] * psi_d[xn][y][zn] * ey[15] +
               w[16] * psi_d[xn][y][zp] * ey[16] +
               w[17] * psi_d[x][yn][zn] * ey[17] +
               w[18] * psi_d[x][yn][zp] * ey[18]);

          force_d[x][y][z][2] =
              -dG * psi_d[x][y][z] *
              (w[1] * psi_d[xp][y][z] * ez[1] + w[2] * psi_d[x][yp][z] * ez[2] +
               w[3] * psi_d[x][y][zp] * ez[3] +
               w[4] * psi_d[xp][yp][z] * ez[4] +
               w[5] * psi_d[xp][yn][z] * ez[5] +
               w[6] * psi_d[xp][y][zp] * ez[6] +
               w[7] * psi_d[xp][y][zn] * ez[7] +
               w[8] * psi_d[x][yp][zp] * ez[8] +
               w[9] * psi_d[x][yp][zn] * ez[9] +
               w[10] * psi_d[xn][y][z] * ez[10] +
               w[11] * psi_d[x][yn][z] * ez[11] +
               w[12] * psi_d[x][y][zn] * ez[12] +
               w[13] * psi_d[xn][yn][z] * ez[13] +
               w[14] * psi_d[xn][yp][z] * ez[14] +
               w[15] * psi_d[xn][y][zn] * ez[15] +
               w[16] * psi_d[xn][y][zp] * ez[16] +
               w[17] * psi_d[x][yn][zn] * ez[17] +
               w[18] * psi_d[x][yn][zp] * ez[18]);
        }
      }
    }
  }
}

//! Update 19 pdfs to the current cell
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
//! \param[in] u_x velocity in x direction
//! \param[in] u_y velocity in y direction
//! \param[in] u_z velocity in z direction
//! \param[in] u_squ sum of squares of u_x, u_y and u_z
__device__ void post_collision(size_t x, size_t y, size_t z, real u_x, real u_y,
                               real u_z, real u_squ) {
  real eu;
  real inv_tau = 1.0 / dtau;
  for (size_t i = 0; i < Q; i++) {
    eu = ex[i] * u_x + ey[i] * u_y + ez[i] * u_z;
    stream_f_d[x][y][z][i] +=
        (t[i] * rho_d[x][y][z] *
             (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_squ) -
         stream_f_d[x][y][z][i]) *
        inv_tau;
  }
}

//! Calculate velocity at current node and transfer updated pdfs to streaming
//! distribution stream_f_d
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
__device__ void compute_velocity(size_t x, size_t y, size_t z) {
  real u_x, u_y, u_z, u_squ;
  real inv_tau = 1.0 / dtau;
  u_x = u_y = u_z = 0.0;
  for (size_t i = 0; i < Q; i++) {
    u_x += stream_f_d[x][y][z][i] * ex[i];
    u_y += stream_f_d[x][y][z][i] * ey[i];
    u_z += stream_f_d[x][y][z][i] * ez[i];
  }

  // <Debut-- Huang>
  u_x += dtau * force_d[x][y][z][0];  ///
  u_y += dtau * force_d[x][y][z][1];  ///
  u_z += dtau * force_d[x][y][z][2];  ///
  // <Fin-- Huang>

  real inv_rho = 1.0 / rho_d[x][y][z];

  u_x *= inv_rho;
  u_y *= inv_rho;
  u_z *= inv_rho;

  u_squ = u_x * u_x + u_y * u_y + u_z * u_z;

  post_collision(x, y, z, u_x, u_y, u_z, u_squ);
}

//! Calculate velocity at current node and transfer updated pdfs to streaming
//! distribution stream_f_d \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
//! \param[in] i i-th velocity direction
__device__ real added_f_at_pressure_bound(size_t x, size_t y, size_t z,
                                          size_t i) {
  real u_x, u_y, u_z, u_squ;
  real eu;
  // real rho_w = 0.1;
  real rho_w = rho_d[x][y][z];
  real inv_tau = 1.0 / dtau;
  u_x = u_y = u_z = 0.0;
  for (size_t i = 0; i < Q; i++) {
    u_x += collision_f_d[x][y][z][i] * ex[i];
    u_y += collision_f_d[x][y][z][i] * ey[i];
    u_z += collision_f_d[x][y][z][i] * ez[i];
  }

  // <Debut-- Huang>
  u_x += dtau * force_d[x][y][z][0];  ///
  u_y += dtau * force_d[x][y][z][1];  ///
  u_z += dtau * force_d[x][y][z][2];  ///
  // <Fin-- Huang>

  real inv_rho = 1.0 / rho_d[x][y][z];

  u_x *= inv_rho;
  u_y *= inv_rho;
  u_z *= inv_rho;

  u_squ = u_x * u_x + u_y * u_y + u_z * u_z;
  eu = ex[i] * u_x + ey[i] * u_y + ez[i] * u_z;

  return 2 * t[i] * rho_w * (1 + 4.5 * eu * eu - 1.5 * u_squ);
}

//! Stream f from the adjacent cells to the current cell in the direction of e
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
//! \param[in] i each for-loop starts from i-th direction at a single thread
//! \param[in] i_max  each for-loop ends at i_max-th direction at a single
//! thread
__device__ void stream_from_neighbor(size_t x, size_t y, size_t z, size_t i,
                                     size_t i_max) {
  size_t next_x, next_y, next_z;
  for (; i < i_max; i++) {
    next_x = x - ex[i];
    if (x == 0 && ex[i] == 1) next_x = lx - 1;
    if (x == lx - 1 && ex[i] == -1) next_x = 0;

    next_y = y - ey[i];
    if (y == 0 && ey[i] == 1) next_y = ly - 1;
    if (y == ly - 1 && ey[i] == -1) next_y = 0;

    next_z = z - ez[i];
    if (z == 0 && ez[i] == 1) next_z = lz - 1;
    if (z == lz - 1 && ez[i] == -1) next_z = 0;

    if (is_fluid_d[next_x][next_y][next_z])
      stream_f_d[x][y][z][i] = collision_f_d[next_x][next_y][next_z][i];
  }
}

//! Apply specified boundary conditions at outer boundary, i.e. bounce_back
//! condition, pressure boundary, and open boundary
//! \param[in] x x-coordinate of center of the sphere
//! \param[in] y y-coordinate of center of the sphere
//! \param[in] z z-coordinate of center of the sphere
__device__ void outer_boundary(size_t x, size_t y, size_t z, size_t i,
                               bool bounce_back, bool pressure_condition,
                               bool open_condition, bool peridoc_condition) {
  size_t switched_i = (i > half) ? (i - half) : (i + half);
  stream_f_d[x][y][z][i] =
      collision_f_d[x][y][z][switched_i] * (bounce_back - pressure_condition) +
      added_f_at_pressure_bound(x, y, z, i) * pressure_condition +
      collision_f_d[x][y][z][i] * open_condition +
      stream_f_d[x][y][z][i] * peridoc_condition;
}

#endif  // ITERATION_DEVICE_CUH_
