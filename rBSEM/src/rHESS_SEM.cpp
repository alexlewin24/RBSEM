// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]

#include <RcppArmadillo.h>
#include "run_HESS.h"

//' @title rHESS_SEM
//' @description
//' Run a simple SEM Bayesian sampler
//' IMPORTANT NOTE: outFilePath must exists, otherwise no output is going to be written.
//' You can make sure of its existence by using base::dir.create(outFilePath) from R
//' @name rHESS_SEM
//' @param inFile path to data file
//' @param outFilePath path to where the output is to be written
//' @param nIter number of iterations
//' @param nChains number of parallel chains to run
//' @param seed pRNG seed
//' @param method $gamma$ sampling method, where 0=$MC^3$ and 1=Thompson -sampling-inspired novel method
//' @examples
//' rHESS_SEM(inFile="Data/sem_data.txt",outFilePath="Data/",nIter=200)
//' 
//' @export
// [[Rcpp::export]]
int rHESS_SEM(std::string inFile, std::string outFilePath, unsigned int nIter=10, 
               unsigned int nChains=1, long long unsigned int seed = 0, unsigned int method = 0) {

	int status = run_HESS(inFile, outFilePath, nIter, nChains, seed, method);
	
	// Exit
	return status;
}
