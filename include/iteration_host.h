#ifndef ITERATION_HOST_H_
#define ITERATION_HOST_H_
#include "lbm_model.cuh"

//! Checks the returned cudaError_t and prints corresponding error message
#define cudaErrorCheck(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

//! Copy constant from the host to the device
//! \param[in] tau Relaxation time
//! \param[in] G The parameter that controls the interaction strength
//! \param[in] carn_star if true C-S EOS is used, else Shan-Chen EOS
//!\param[in] T_Tc T over Tc ratio
//! \param[in] rhol_spinodal rho used to identify the liquid nodes
//! \param[in] rhog_spinodal rho used to identify the vapor nodes
//! \param[in] inject_density If density is injected
//! \param[in] rho_inject_period The density injection/drainage period
//! \param[in] rho_increment The density injection/drainage increment
//! \param[in] rhos The density of solids (controls the contact angle)
//! \param[in] inject_position The injection/drainage positions
//! \param[in] statPeriod The step increment for outputting model statistics
void transfer_parameter(real* tau, real* G, bool* carn_star, real* T_Tc,
                        real* rhol_spinodal, real* rhog_spinodal,
                        bool* inject_density, size_t* rho_inject_period,
                        real* rho_increment, real* rhos, real* inject_position,
                        size_t* statPeriod);

//! Copy everything from fields on DRAM in GPU
//! \param[in] rho_dd The address of density
//! \param[in] psi_dd The address of pseudopotential
//! \param[in] pressure_dd The address of pressure
//! \param[in] is_fluid_dd The address of the is_fluid
//! \param[in] collision_f_dd The address of colliding pdfs
//! \param[in] stream_f_dd The address of streaming pdfs
//! \param[in] force_dd The address of force

void multiphaselbm_initialization(Realxyz** rho_dd, Realxyz** psi_dd,
                                  Realxyz** pressure_dd, Boolxyz** is_fluid_dd,
                                  RealxyzQ** collision_f_dd,
                                  RealxyzQ** stream_f_dd, Realxyz3** force_dd);

//! 3D multiphaseLBM solver on GPU
//! \param[in] step Current timestep at which the next iteration begins
//! \param[in] statPeriod The step increment for outputting model statistics
//! \param[in] reverse If reverse=1 inject else if reverse=-1 drain
//! \param[in] average_liquid_pressure Averaged liquid pressure output to the
//! host \param[in] average_vapor_pressure Averaged vapor pressure output to the
//! host \param[in] saturation_ratio Saturation ratio output to the host
//! \param[in] average_liquid_density Averaged liquid density output to the host
//! \param[in] average_vapor_density Averaged vapor density output to the host
void multiphaselbm_iteration(long step, size_t statPeriod, int reverse,
                             unsigned int inject_type);

#endif  // ITERATION_HOST_H_
