// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]

#include <RcppArmadillo.h>
#include "run_HESS.h"

//' @title rHESS_SEM_internal
//' @description
//' Run a simple SEM Bayesian sampler -- internal function
//' @name rHESS_SEM_internal
//' @param inFile path to data file
//' @param outFilePath path to where the output is to be written
//' @param nIter number of iterations
//' @param nChains number of parallel chains to run
//' @param seed pRNG seed
//' @param method \deqn{\gamma}{gamma} sampling method, where 0=\deqn{MC^2}{MC^3} and 1=Thompson -sampling-inspired novel method
//' @param writeOutputLevel 1=no imputed values output to file, 2=imputed values are output to file (future implement 0 for only posterior means output)
//
// NOTE THAT THIS IS BASICALLY JUST A WRAPPER

// [[Rcpp::export]]
int rHESS_SEM_internal(std::string inFile, std::string outFilePath, bool autoAddIntercept,
                       std::string gammaInit = "S", unsigned int nIter=10, unsigned int burnin = 0, unsigned int nChains=1, 
                       unsigned long long seed = 0, int method = 0, int writeOutputLevel = 1){
  
  int status = run_HESS(inFile, outFilePath, autoAddIntercept, gammaInit, nIter, burnin, nChains, seed, method, writeOutputLevel);
  
  // Exit
  return status;
  
}