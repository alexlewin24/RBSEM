
#ifndef RUN_HESS
#define RUN_HESS

#include <vector>
#include <iostream>
#include <string>
// #include <RcppArmadillo.h>
#include <armadillo>

#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "global.h"
#include "utils.h"
#include "distr.h"
#include "imputation.h"
#include "HESS.h"

#ifdef _OPENMP
extern omp_lock_t RNGlock; //defined in global.h
#endif
extern std::vector<std::mt19937_64> rng;

int run_HESS(std::string inFile, std::string outFilePath, 
             bool autoAddIntercept, std::string gammaInit,
             unsigned int nIter, unsigned int burning, unsigned int nChains,
             unsigned long long seed, int method, int writeOutputLevel);

#endif