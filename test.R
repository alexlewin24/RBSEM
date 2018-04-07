#install.packages(c("Rcpp","RcppArmadillo","BH"))
library(Rcpp)
library(RcppArmadillo)
library(BH)

Sys.setenv("CXXFLAGS"=paste(sep="","-I'",getwd(),"/'"))
sourceCpp("r_HESS.cpp", rebuild=TRUE) #, verbose = TRUE

# seems to compile all right the dependencies, but somehow rng is failing

