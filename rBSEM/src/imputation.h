#ifndef IMPUTATION
#define IMPUTATION

#include "global.h"
#include "utils.h"
#include "distr.h"

class Imputation{

    public:
    Imputation(arma::mat& data, const arma::uvec& completeCases, const arma::umat& SEMGraph, const std::vector<arma::uvec>& blockIdx, const arma::ivec& varType);
    
    void imputeAll(arma::mat& data, const arma::uvec& missingDataIndexes, const arma::umat& missingDataIdxArray, arma::ivec& varType,
        const std::vector<arma::uvec>& outcomesIdx, const std::vector<arma::uvec>& fixedPredictorsIdx, 
        const std::vector<arma::uvec>& vsPredictorsIdx,
        const std::vector<arma::ucube>& gamma_state, const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0 );

    private:

    	arma::vec covariatesOnlyMean;
        arma::mat covariatesOnlyVar;
	    arma::uvec covariatesOnlyIdx;
        size_t nCompleteCases;

};

#endif