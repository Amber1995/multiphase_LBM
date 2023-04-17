#ifndef LBM_MODEL_CUH_
#define LBM_MODEL_CUH_
#define Q 19
#define lx 400
#define ly 400
#define lz 400
#include <stdio.h>

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

using real = float;

//! Define types of pointers to 3/4d array
typedef real Realxyz3[ly][lz][3];
typedef real RealxyzQ[ly][lz][Q];
typedef real Realxyz[ly][lz];
typedef size_t Size_txyz[ly][lz];
typedef bool Boolxyz[ly][lz];

const int half = (Q - 1) / 2;

// Block and grid sizes
const int BLKXSIZE = 4;
const int BLKYSIZE = 4;
const int BLKZSIZE = 4;
// Number of grids in x direction (in order to accelerate streaming)
// (Q-1)=18 is divisible by SCALE_LX, so SCALE_LX can be 1, 2, 3, 6, 9 or 18
const int SCALE_LX = 3;

//! Definition of LBM weights
const real t0 = 1.0 / 3.0;
const real t1 = 1.0 / 18.0;
const real t2 = 1.0 / 36.0;
CONSTANT real t[Q] = {t0, t1, t1, t1, t2, t2, t2, t2, t2, t2,
                      t1, t1, t1, t2, t2, t2, t2, t2, t2};

//! Definition of Shan-Chen factors for force computation
const int w0 = 0;
const int w1 = 2;
const int w2 = 1;
CONSTANT real w[Q] = {t0, t1, t1, t1, t2, t2, t2, t2, t2, t2,
                      t1, t1, t1, t2, t2, t2, t2, t2, t2};

//! x component of predefined velocity in Q directions
CONSTANT int ex[Q] = {0, -1, 0, 0, -1, -1, -1, -1, 0, 0,
                      1, 0,  0, 1, 1,  1,  1,  0,  0};
CONSTANT int ey[Q] = {0, 0, -1, 0, -1, 1, 0, 0, -1, -1,
                      0, 1, 0,  1, -1, 0, 0, 1, 1};
CONSTANT int ez[Q] = {0, 0, 0, -1, 0, 0, -1, 1, -1, 1,
                      0, 0, 1, 0,  0, 1, -1, 1, -1};

//! Convert a real number to the nearest integer
//! \param[in] x The real value to be rounded to the nearest integer
inline int nearest(real x) { return (int)floor(x + 0.5); }

#endif  // LBM_MODEL_CUH_
