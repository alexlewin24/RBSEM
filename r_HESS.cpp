// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]


// [[Rcpp::depends(BH)]]

#include <vector>
#include <iostream>
#include <string>
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

// [[Rcpp::export]]
void HESS_SEM_Reg(std::string inFile, std::string outFilePath, unsigned int nIter, unsigned int nChains, long long unsigned int seed = 0, int method=1)
{

	omp_init_lock(&RNGlock);  // RNG lock for the parallel part

  if( inFile == "" ){
    inFile = "data.txt"; // just to protect against malicious empty strings
  }

	// ############# Read the data
	unsigned int n,p;
	arma::uvec s,NAIdx;
	arma::ivec blockLabel,varType;
	arma::mat data;

	if( Utils::readDataSEM(inFile, data, blockLabel, varType, NAIdx,
		s, p, n) ){
		std::cout << "Reading successfull!" << std::endl;
	}else{
		std::cout << "OUCH! EXITING --- " << std::endl;
		return;
	}

	unsigned int nBlocks = s.n_elem;

	// Add to X the intercept
	arma::uword tmpUWord = arma::as_scalar(arma::find(blockLabel == 0,1,"first"));
	data.insert_cols( tmpUWord , arma::ones<arma::vec>(n) );
	blockLabel.insert_cols( tmpUWord, arma::zeros<arma::ivec>(1) ); // add its blockLabel
	varType.insert_cols( tmpUWord, arma::zeros<arma::ivec>(1) ); // add its type

	// now find the indexes for each block in a more armadillo-interpretable way
	std::vector<arma::uvec> blockIdx(nBlocks+1);
	for( unsigned int k = 0; k<(nBlocks+1); ++k)
	{
		blockIdx[k] = arma::find(blockLabel == 0);
	}

  // ############# Init the RNG generator/engine
	unsigned int nThreads = omp_get_max_threads();
	if(nChains < nThreads)
		nThreads = nChains;
	omp_set_num_threads(nThreads);

	rng.reserve(nThreads);  // reserve the correct space for the vector of rng engines
  
	// seed all the engines
	// for(unsigned int i=0; i<nThreads; ++i)
	// {
  //   rng[i] = std::mt19937_64(  seed + i*(1000*(p*s*3+s*s)*nIter) );
	// 	// rng[i].seed( ); // 1000 rng per (p*s*3+s*s) variables each loop .. is this...ok? is random better?
	// }




}