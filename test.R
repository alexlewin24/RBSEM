library(pkgKitten)

kitten(name = "rBSEM")
# R CMD build
# R CMD check --as.cran

#install.packages(c("devtools","Rcpp","RcppArmadillo","roxygen2","BH"))
library(devtools)

## --------------
setwd("rBSEM/")
use_rcpp()
## follow instructions


## --------------
