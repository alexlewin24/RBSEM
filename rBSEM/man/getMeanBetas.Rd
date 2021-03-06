% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/getMeanBetas.R
\name{getMeanBetas}
\alias{getMeanBetas}
\title{Get the conditional posterior mean regression coefficients}
\usage{
getMeanBetas(
  blockGraph,
  blockList,
  outFilePath,
  outFilePrefix,
  varNames,
  Intercept = TRUE
)
}
\arguments{
\item{blockGraph}{adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
blockGraph should be a square matrix (adjacency for blocks)
columns are LHS of regressions
rows are input RHS of regressions}

\item{blockList}{list of variables in each block (as input to \code{\link{rHESS_SEM}})}

\item{outFilePath}{filepath for RBSEM output files}

\item{outFilePrefix}{file prefix for RBSEM output files (this will be the name of the input data file, stripped of any .txt)}

\item{varNames}{vector of variable names from original input data file}

\item{Intercept}{does the model include intercept terms? (default TRUE)}
}
\value{
Matrix of posterior mean regression coefficients 
(conditional on being selected in the model) in adjacency matrix format.
Columns are response variables, rows are covariates.
}
\description{
This function does 2 things:
prints individual small matrices of betas for all regressions (beta.postmean)
returns the large beta matrix for all variables together (betaAdjgraph)

The output is a square matrix with dimensions m x m where m is the total number 
of variables input into the model. Element (j,k) contains the posterior mean 
of the regression coefficient for covariate j in the regression for 
response variable k, conditional on that covariate being selected into the
regression. The unconditional (shrunk) coefficients can be found by multiplying
by the posterior inclusion probability (see \code{\link{getMeanGammas}}).

There may also be covariates which are forced into some regressions: these
will have betas but not gammas. 

The output beta matrix does not include the intercept terms from the regression.
However these are printed as part of the small matrices for the individual 
regressions.
}
\examples{
require(utils)
dir.create("tmp")
data(sample_SEM)
write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
blockL = list( 
  c(9:28),  ## x0 -- block 0
  c(1:5),  ## y1 -- block 1
  c(6:8)  ## y2 -- block 2
)
G = matrix(c( 
  0,1,1,
  0,0,1,
  0,0,0 ), 
  byrow=TRUE,ncol=3,nrow=3)
rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
         SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0,nChains = 2)
getMeanBetas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
         outFilePrefix="sem_data",varNames=names(sample_SEM))
unlink("tmp", recursive=TRUE)
 
}
