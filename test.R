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
# each time you modify the source code do
# R CMD build rBSEM
#  in R > devtools::document("rBSEM/")
# R CMD check --as-cran rBSEM_1.0.tar.gz
# then for both u and the users
#  in R > install.packages("rBSEM_1.0.tar.gz", repos = NULL, type="source")
#  in R > library('rBSEM')
#  in R > rHESS_SEM(inFile="Data/sem_data.txt",outFilePath="Data/",nIter=20000)



