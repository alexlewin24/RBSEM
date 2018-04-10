// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// rHESS_SEM
int rHESS_SEM(std::string inFile, std::string outFilePath, unsigned int nIter, unsigned int nChains, long long unsigned int seed, unsigned int method);
RcppExport SEXP _rBSEM_rHESS_SEM(SEXP inFileSEXP, SEXP outFilePathSEXP, SEXP nIterSEXP, SEXP nChainsSEXP, SEXP seedSEXP, SEXP methodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type inFile(inFileSEXP);
    Rcpp::traits::input_parameter< std::string >::type outFilePath(outFilePathSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nIter(nIterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nChains(nChainsSEXP);
    Rcpp::traits::input_parameter< long long unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type method(methodSEXP);
    rcpp_result_gen = Rcpp::wrap(rHESS_SEM(inFile, outFilePath, nIter, nChains, seed, method));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rBSEM_rHESS_SEM", (DL_FUNC) &_rBSEM_rHESS_SEM, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_rBSEM(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
