#' @title Get Layers (Blocks) for an input vector of variables (internal function) 

#' @description 
#' This function takes as input a vector of variable names, and the blockList 
#' of the model. The function outputs the block number for each variable.
#' This is an internal function used when plotting layered graphs and subgraphs.
#' 
#' @param variables vector of variables
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' 
#' @examples 
#' require(utils)
#' dir.create("tmp")
#' data(sample_SEM)
#' write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
#' blockL = list( 
#'   c(9:28),  ## x0 -- block 0
#'   c(1:5),  ## y1 -- block 1
#'   c(6:8)  ## y2 -- block 2
#' )
#' G = matrix(c( 
#'   0,1,1,
#'   0,0,1,
#'   0,0,0 ), 
#'   byrow=TRUE,ncol=3,nrow=3)
#'   
#' ## create input adjacency matrix for variables
#' inputAdj <- getInputGraph(blockList = blockL,blockGraph = G,varNames=names(sample_SEM))
#' ## plot input graph with all variables
#' plotLayeredGraph(inputAdj,blockList = blockL)
#'
#' ## fit the Bayesian model
#' rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
#'          SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0,nChains = 2)
#'          
#' ## get the marginal probabilities of inclusion (mPPIs)         
#' mPPIs <- getMeanGammas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'          outFilePrefix="sem_data",varNames=names(sample_SEM))
#' ## plot the output graph, thresholding mPPIs at 0.5
#' plotLayeredGraph(mPPIs>=0.5,blockList = blockL)
#'          
#' unlink("tmp", recursive=TRUE)
#'  
#' @return vector of block numbers (layers), same length as input vector of variables
#' @export

getLayersfromBlockList <- function(variables,blockList,varNames){
  
  varIndices <- match(x=variables,table=varNames)
  #print(varIndices)
  
  # this code from stackoverflow
  g <- rep(seq_along(blockList), sapply(blockList, length))
  #print(g)
  layers <- g[match(varIndices, unlist(blockList))]
  #print(layers)
  return(layers)
  
}


