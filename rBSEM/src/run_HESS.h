
#ifndef RUN_HESS
#define RUN_HESS

#include <vector>
#include <iostream>
#include <string>
// #include <RcppArmadillo.h>
#include <armadillo>

#include <cmath>
#include <limits>
#include <omp.h>

#include "global.h"
#include "utils.h"
#include "distr.h"
#include "imputation.h"
#include "HESS.h"

extern omp_lock_t RNGlock; //defined in global.h
extern std::vector<std::mt19937_64> rng;

int run_HESS(std::string inFile, std::string outFilePath, bool autoAddIntercept,
             unsigned int nIter, unsigned int nChains,
             unsigned long long seed, int method);

#endif