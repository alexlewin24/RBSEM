#ifndef IMPUTATION
#define IMPUTATION

#include "global.h"
#include "utils.h"
#include "distr.h"

namespace Imputation{

    void initialiseXimpute(arma::mat& data, arma::uvec& missingDataIndexes, arma::umat& SEMGraph, std::vector<arma::uvec>& blockIdx,
        arma::vec& covariatesOnlyMean, arma::vec& covariatesOnlyVar, arma::uvec& covariatesOnlyIdx);

    void imputeAll(arma::mat& data, const arma::uvec& missingDataIndexes, const arma::umat& missingDataIdxArray, arma::ivec& varType,
        const arma::vec& covariatesOnlyMean, const arma::vec& covariatesOnlyVar, const arma::uvec& covariatesOnlyIdx,
        const std::vector<arma::uvec>& outcomesIdx, const std::vector<arma::uvec>& fixedPredictorsIdx, 
        const std::vector<arma::uvec>& vsPredictorsIdx,
        const std::vector<arma::ucube>& gamma_state, const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0 );

}

#endif