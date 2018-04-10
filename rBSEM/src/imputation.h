#include "global.h"
#include "utils.h"
#include "distr.h"

#ifndef IMPUTATION
#define IMPUTATION

void imputeX( arma::mat &X , arma::uvec& missingDataIndexes );
void imputeY( arma::mat &Y , arma::mat &X , arma::uvec& missingDataIndexes , arma::umat &gamma );

#endif