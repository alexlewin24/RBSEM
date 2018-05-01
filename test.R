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
devtools::has_devel()
devtools::document("rBSEM")
# devtools::test(pkg = "rBSEM")
# devtools::check("rBSEM", cran=TRUE)
devtools::build("rBSEM")
remove.packages("rBSEM")
install.packages("rBSEM_0.1.0.tar.gz", repos = NULL, type="source")


load("data/sample_data.RData")
# C Primer
# source("testCpp.R"); cppPrimer(inFile="data/sem_data.txt",blockList = blockL,SEMGraph = G,outFilePath="data/")

rBSEM::rHESS_SEM(inFile="data/sem_data.txt",blockList = blockL,
                 SEMGraph = G,outFilePath="data/",nIter=50000,method = 1)

# rBSEM::rHESS_SEM(inFile="data/sem_data.txt",blockList = blockL,
#                  SEMGraph = G,outFilePath="data/",nIter=2000, nChains = 2)


## then check some output
greyscale = grey((0:1000)/1000)

est_gamma_1 = as.matrix( read.table("data/sem_data_HESS_gamma_0_out.txt") )
est_gamma_2 = as.matrix( read.table("data/sem_data_HESS_gamma_1_out.txt") )

par(mfrow=c(2,2))
image(est_gamma_1,col=greyscale); image(gamma_1[-1,],col=greyscale)
image(est_gamma_2,col=greyscale); image(gamma_2[-1,],col=greyscale)
par(mfrow=c(1,1))

