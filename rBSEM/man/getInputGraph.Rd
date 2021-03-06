% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/getInputGraph.R
\name{getInputGraph}
\alias{getInputGraph}
\title{Creates the maximal adjacency matrix allowed for the variables input into the model.}
\usage{
getInputGraph(blockGraph, blockList, varNames = NULL)
}
\arguments{
\item{blockGraph}{adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
blockGraph should be a square matrix (adjacency for blocks)
columns are LHS of regressions
rows are input RHS of regressions}

\item{blockList}{list of variables in each block (as input to \code{\link{rHESS_SEM}})}

\item{varNames}{vector of variable names from original input data file}
}
\value{
Adjacency Matrix of input model
}
\description{
The rBSEM package paramterises the Bayesian SEM model through a DAG
based on the blocks, plus a list of variables for each block.

This function expands the input DAG between blocks
of variables to give an adjacency matrix between variables. 
It is useful if you want to plot a DAG showing which input variables you
are allowing into the model (before fitting the model).
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
getInputGraph(blockList = blockL,blockGraph = G,varNames=names(sample_SEM))
unlink("tmp", recursive=TRUE)
 
}
