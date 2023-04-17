# Multiphase LBM-DEM

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/cb-geo/multiphase-lbm/develop/license.md)
[![CircleCI](https://circleci.com/gh/cb-geo/multiphase-lbm.svg?style=svg)](https://circleci.com/gh/cb-geo/multiphase-lbm)


## Compile

0. Run `mkdir build && cd build && cmake ..`.

1. Run `make clean && make -jN` (where N is the number of cores).

## Run

0. Run `./multiphase /<path-to-inputfile>/inputfile.txt`. For the input file, follow the format of `example.txt` provided.


## TACC Maverick 2

Load required modules

```
module load cmake/3.16.1
module load gcc/7.3.0
module load cuda/11.0
```

Compile as per the instructions above.
