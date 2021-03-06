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
# devtools::use_vignette("rBSEM-vignette",pkg = "rBSEM")


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


library(devtools)
# better build routine
devtools::has_devel()
# devtools::test(pkg = "rBSEM")
# devtools::check("rBSEM", cran=TRUE)

remove.packages("rBSEM")
Rcpp::compileAttributes(pkgdir = "rBSEM/"); devtools::document("rBSEM");
devtools::build("rBSEM",vignettes=TRUE)
install.packages("rBSEM_0.2.tar.gz", repos = NULL, type="source",)
# devtools::install_local("rBSEM_0.2.tar.gz",build_vignettes = TRUE,force=TRUE) # this one forces to build vignettes

# C Primer
# source("testCpp.R"); cppPrimer(inFile="data/sem_data.txt",blockList = blockL,SEMGraph = G,outFilePath="data/")

# browseVignettes("rBSEM")

na = TRUE
nIter = 10000 * 1.1
burnin = 0.09 * nIter

if(!na){
  load("data/sample_data.RData")
  rBSEM::rHESS_SEM(inFile="data/sem_data.txt",blockList = blockL,
                   SEMGraph = G,outFilePath="data/",nIter=nIter, burnin=burnin, method=1, nChains = 4)
}else{
  load("data/na_sample_data.RData")
  rBSEM::rHESS_SEM(inFile="data/na_sem_data.txt",blockList = blockL,
                   SEMGraph = G,outFilePath="data/",nIter=nIter, burnin=burnin, method=1, nChains = 4)
}


## then check some output
greyscale = grey((0:100)/100)

if(!na){
  est_gamma_1 = as.matrix( read.table("data/sem_data_HESS_gamma_1_out.txt") )
  est_gamma_2 = as.matrix( read.table("data/sem_data_HESS_gamma_2_out.txt") )
  
  est_beta_1 = as.matrix( read.table("data/sem_data_HESS_beta_1_out.txt") )
  est_beta_2 = as.matrix( read.table("data/sem_data_HESS_beta_2_out.txt") )
  
}else{
  est_gamma_1 = as.matrix( read.table("data/na_sem_data_HESS_gamma_1_out.txt") )
  est_gamma_2 = as.matrix( read.table("data/na_sem_data_HESS_gamma_2_out.txt") )
  
  est_beta_1 = as.matrix( read.table("data/na_sem_data_HESS_beta_1_out.txt") )
  est_beta_2 = as.matrix( read.table("data/na_sem_data_HESS_beta_2_out.txt") )
}

par(mfrow=c(2,2))
image(est_gamma_1,col=greyscale); image(gamma_1[-1,],col=greyscale)
image(est_gamma_2,col=greyscale); image(gamma_2[-1,],col=greyscale)
par(mfrow=c(1,1))


par(mfrow=c(2,2))
image(est_beta_1,col=greyscale); image(b_1*gamma_1,col=greyscale)
image(est_beta_2,col=greyscale); image(b_2*gamma_2,col=greyscale)
par(mfrow=c(1,1))

if( burnin < 1 ) burnin = ceiling(nIter * burnin)
if(!na){
  plot(scan("data/sem_data_HESS_logP_out.txt"),type="l")
}else{
  plot(scan("data/na_sem_data_HESS_logP_out.txt"),type="l")
}

