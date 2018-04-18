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
# R CMD check --as-cran rBSEM_0.1.0.tar.gz   ## or devtools::check("rBSEM/",cran = TRUE)
# then for both u and the users
# Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
# in R > install.packages("rBSEM_0.1.0.tar.gz", repos = NULL, type="source")
#  in R > library('rBSEM')
#  in R > rHESS_SEM(inFile="Data/sem_data.txt",outFilePath="Data/",nIter=20000)



# better build rutine

devtools::document("rBSEM")
devtools::check("rBSEM", cran=TRUE)
devtools::build("rBSEM")
remove.packages("rBSEM")
install.packages("rBSEM_0.1.0.tar.gz", repos = NULL, type="source")
# devtools::test(pkg = "rBSEM")
rBSEM::rHESS_SEM_internal(inFile="data/sem_data.txt",outFilePath="data/",nIter=2000)
rBSEM::rHESS_SEM(inFile="data/sem_data.txt",outFilePath="data/",nIter=2000)



