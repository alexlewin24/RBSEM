#install.packages(c("Rcpp","RcppArmadillo","BH"))
library(Rcpp)
library(RcppArmadillo)
library(BH)

sourceCpp("r_HESS.cpp", rebuild=TRUE, verbose = TRUE)
