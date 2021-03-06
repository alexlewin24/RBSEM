% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/getCredIntBetas.R
\name{getCredIntBetas}
\alias{getCredIntBetas}
\title{Get posterior credible intervals for regression coefficients}
\usage{
getCredIntBetas(
  blockGraph,
  blockList,
  outFilePath,
  outFilePrefix,
  varNames,
  Intercept = TRUE,
  probCI
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

\item{probCI}{the probability to be contained with the credible intervals}
}
\value{
None
}
\description{
This function loops over all the multivariate regressions, and for each
regression coefficient calculates a Highest Posterior Density credible
interval (using the \code{\link{coda}} package).
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
getCredIntBetas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
         outFilePrefix="sem_data",varNames=names(sample_SEM),probCI=0.9)
unlink("tmp", recursive=TRUE)

}
