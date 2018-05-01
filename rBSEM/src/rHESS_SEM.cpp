// [[Rcpp::plugins(cpp11)]]
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
//' @examples
//' dir.create("tmp") 
//' data(sample_SEM)
//' write.table(sample_sem,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
//' 
//' blockList = list(c(1),c(rep(0,ncol(sample_SEM)-1)))
//' varType = rep(0,ncol(sample_SEM))
//' rHESS_SEM_internal(inFile="tmp/sem_data.txt",outFilePath="tmp/",nIter=200)
//' unlink("tmp", recursive=TRUE)
//
// NOTE THAT THIS IS BASICALLY JUST A WRAPPER

// [[Rcpp::export]]
int rHESS_SEM_internal(std::string inFile, std::string outFilePath, bool autoAddIntercept,
                       unsigned int nIter=10, unsigned int nChains=1, 
                       unsigned long long seed = 0, int method = 0){
  
  int status = run_HESS(inFile, outFilePath, autoAddIntercept, nIter, nChains, seed, method);
  
  // Exit
  return status;
  
  // 	return Rcpp::List::create(Named("coef") = coef,
  //                      Named("sderr")= sderr);
}