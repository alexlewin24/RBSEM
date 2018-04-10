// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]


// [[Rcpp::depends(BH)]]

#include "run_HESS.h"

// [[Rcpp::export]]
int HESS_SEM_Reg(std::string inFile, std::string outFilePath, unsigned int nIter, unsigned int nChains, long long unsigned int seed = 0, int method=1)
{

	omp_init_lock(&RNGlock);  // RNG lock for the parallel part

	if( inFile == "" ){
		inFile = "data.txt"; // just to protect against malicious empty strings
	}

	int status = run_HESS(inFile, outFilePath, nIter, nChains, seed, method);

	// Exit
	return 0;

}