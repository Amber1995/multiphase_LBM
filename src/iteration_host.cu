#include "iteration_host.h"
#include "iteration_kernel.cuh"
#include "lbm_model.cuh"

const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
const dim3 gridSize(((lx + BLKXSIZE - 1) / BLKXSIZE),
                    ((ly + BLKYSIZE - 1) / BLKYSIZE),
                    ((lz + BLKZSIZE - 1) / BLKZSIZE));

//! Invoke double grids to excute the streaming
const dim3 gridSize_Q(((SCALE_LX * lx + BLKXSIZE - 1) / BLKXSIZE),
                      ((ly + BLKYSIZE - 1) / BLKYSIZE),
                      ((lz + BLKZSIZE - 1) / BLKZSIZE));

//! Copy constant parameters to global symbols
void transfer_parameter(real* tau, real* G, bool* carn_star, real* T_Tc,
                        real* rhol_spinodal, real* rhog_spinodal,
                        bool* inject_density, size_t* rho_inject_period,
                        real* rho_increment, real* rhos, real* inject_position,
                        size_t* statPeriod) {

  cudaErrorCheck(cudaMemcpyToSymbol(dtau, tau, sizeof(real)));
  cudaErrorCheck(cudaMemcpyToSymbol(dG, G, sizeof(real)));
  cudaErrorCheck(cudaMemcpyToSymbol(dcarn_star, carn_star, sizeof(bool)));
  cudaErrorCheck(cudaMemcpyToSymbol(dT_Tc, T_Tc, sizeof(real)));
  cudaErrorCheck(
      cudaMemcpyToSymbol(drhol_spinodal, rhol_spinodal, sizeof(real)));
  cudaErrorCheck(
      cudaMemcpyToSymbol(drhog_spinodal, rhog_spinodal, sizeof(real)));
  cudaErrorCheck(
      cudaMemcpyToSymbol(dinject_density, inject_density, sizeof(bool)));
  cudaErrorCheck(cudaMemcpyToSymbol(drho_inject_period, rho_inject_period,
                                    sizeof(size_t)));
  cudaErrorCheck(
      cudaMemcpyToSymbol(drho_increment, rho_increment, sizeof(real)));
  cudaErrorCheck(cudaMemcpyToSymbol(drhos, rhos, sizeof(real)));
  cudaErrorCheck(cudaMemcpyToSymbol(dinject_position, inject_position,
                                    sizeof(unsigned int)));
  cudaErrorCheck(cudaMemcpyToSymbol(dstatPeriod, statPeriod, sizeof(size_t)));
  cudaDeviceSynchronize();
}

//! Initialize multidimensional arrays on GPU
void multiphaselbm_initialization(Realxyz** rho_dd, Realxyz** psi_dd,
                                  Realxyz** pressure_dd, Boolxyz** is_fluid_dd,

                                  RealxyzQ** collision_f_dd,
                                  RealxyzQ** stream_f_dd, Realxyz3** force_dd) {

  cudaErrorCheck(cudaMemcpyToSymbol(rho_d, rho_dd, sizeof(*rho_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(psi_d, psi_dd, sizeof(*psi_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(pressure_d, pressure_dd,
                                    sizeof(*pressure_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(is_fluid_d, is_fluid_dd,
                                    sizeof(*is_fluid_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(collision_f_d, collision_f_dd,
                                    sizeof(*collision_f_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(stream_f_d, stream_f_dd,
                                    sizeof(*stream_f_dd), 0,
                                    cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(force_d, force_dd, sizeof(*force_dd), 0,
                                    cudaMemcpyHostToDevice));

  multiphaselbm_init<<<gridSize, blockSize, 0>>>();
}

//! Invoke kernel functions to run iterations on GPU
void multiphaselbm_iteration(long step, size_t statPeriod, int reverse,
                             unsigned int inject_type) {
  // The inject type can be changed under given condition
  dinject_type = inject_type;
  do_streaming<<<gridSize_Q, blockSize, 0>>>();
  treat_boundary<<<gridSize, blockSize, 0>>>();
  do_collision<<<gridSize, blockSize, 0>>>(step, reverse);
  do_swap<<<1, 1, 0>>>();
}
