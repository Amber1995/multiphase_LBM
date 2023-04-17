#ifndef GLOBAL_VAR_CUH_
#define GLOBAL_VAR_CUH_
#include "lbm_model.cuh"

//! Arrays updated in iterations in unified memory
// Density
__device__ __managed__ Realxyz* rho_d;
// Pressure
__device__ __managed__ Realxyz* pressure_d;
// If true the current node is fluid node
__device__ __managed__ Boolxyz* is_fluid_d;

//! Arrays updated in iterations in global memory
// pseudopotential
__device__ Realxyz* psi_d;
// Distributions updated in collision
__device__ RealxyzQ* collision_f_d;
// Distributions updated in streaming
__device__ RealxyzQ* stream_f_d;
// Force
__device__ Realxyz3* force_d;

//! Define constant parameters in global memory
//! \details Parameters are copied from the host distinguished by initial d's
__device__ real dtau;
__device__ real dG;
__device__ bool dcarn_star;
__device__ real dT_Tc;
__device__ real drhol_spinodal;
__device__ real drhog_spinodal;
__device__ bool dinject_density;
__device__ size_t drho_inject_period;
__device__ real drho_increment;
__device__ real drhos;
__device__ __managed__ unsigned int dinject_type;
__device__ real dinject_position;
__device__ size_t dstatPeriod;

__device__ bool periodic_x = true;
__device__ bool periodic_y = true;
__device__ bool periodic_z = true;

// If at least one direction is not periodic
__device__ bool xmin_bounce_back = false;
__device__ bool ymin_bounce_back = false;
__device__ bool zmin_bounce_back = false;
__device__ bool xmax_bounce_back = false;
__device__ bool ymax_bounce_back = false;
__device__ bool zmax_bounce_back = false;

__device__ bool xmin_pressure_condition = false;
__device__ bool ymin_pressure_condition = false;
__device__ bool zmin_pressure_condition = false;
__device__ bool xmax_pressure_condition = false;
__device__ bool ymax_pressure_condition = false;
__device__ bool zmax_pressure_condition = false;

__device__ bool xmin_open_condition = false;
__device__ bool ymin_open_condition = false;
__device__ bool zmin_open_condition = false;
__device__ bool xmax_open_condition = false;
__device__ bool ymax_open_condition = false;
__device__ bool zmax_open_condition = false;

#endif  // GLOBAL_VAR_CUH_
