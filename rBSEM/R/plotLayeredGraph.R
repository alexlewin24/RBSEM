#' @title Plot a Directed Acyclic Graph with variables as nodes. 

#' @description 
#' This function produces a Directed Acyclic Graph (DAG) based on an
#' adjacency matrix (using \code{\link{igraph}} package functions). The DAG layout
#' is a layered graph, where the layers correspond to the SEM blocks 
#' (as input to \code{\link{rHESS_SEM}}).
#' 
#' This function can be used to plot the input model (just displaying the variables
#' in each block and the DAG causal structure assumed), or the output model based
#' on an adjacency matrix found by thresholding the posterior inclusion
#' probabilities. It could also be used to plot a DAG using posterior regression
#' coefficients to define the presence or absence of an edge between nodes. 

#' @param adjMatrix adjacency matrix for variables
#' columns are variables, rows are variables. Element(i,j) of matrix is 1 if there 
#' is an arrow from i to j. This means that for a DAG the matrix should be upper 
#' triangular.
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param varNames vector of variable names from original input data file
#' @param weighted parameter passed to igraph to tell whether graph edges are 
#' weighted (TRUE) or not (FALSE or NULL). Default NULL.
#' @param ... formatting parameters to be passed to igraph plot function 
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
#' 
#' @return None
#' @export
#' @importFrom igraph graph.adjacency degree V delete.vertices plot.igraph

plotLayeredGraph <- function(adjMatrix,blockList,varNames,weighted=NULL,...){
  
  # igraph function convert adjacency matrix to igraph object
  graphAdj <- igraph::graph.adjacency(as.matrix(adjMatrix),diag=FALSE,
                                      mode="directed",weighted=weighted)
  
  variable_layers <- getLayersfromBlockList(row.names(adjMatrix),blockList,varNames)
  #print(variable_layers)
  
  #nblocks <- length(unique(variable_layers))
  nblocks <- length(blockList)
  cat("nblocks =",nblocks,"\n")
  nvar <- dim(adjMatrix)[2]
  cat("nvar =",nvar,"\n")

  # get layout co-ords 
  llcoords <- matrix(rep(0,2*nvar),ncol=2)
  # x co-ord is number of layer
  llcoords[,1] <- variable_layers
    
  #print(llcoords) 
  # loop over blocks to get y co-ords
  for(i in 1:nblocks){
    #llcoords[blockList[[i]],1] <- i-1
    #len <- length(blockList[[i]])
    memblock <- which(variable_layers==i)
    #print(memblock)
    len <- length(memblock)
    llcoords[memblock,2] <- ( 0:(len-1) - (len-1)/2 )*2/len
    #llcoords[blockList[[i]],2] <- ( 0:(len-1) - (len-1)/2 )*2/len
  }
  #print(llcoords)

  
  igraph::V(graphAdj)$layer <- variable_layers
  
  #print(degree(graphAdj))
  #print(llcoords)
  
  # remove unconnected nodes
  layout <- llcoords[igraph::degree(graphAdj)!=0,]
  #print(layout)
  graphAdj <- igraph::delete.vertices(graphAdj,igraph::degree(graphAdj)==0) 
  
  if(is.null(weighted)){
    igraph::plot.igraph(graphAdj,layout=layout, rescale=TRUE, 
                        vertex.color="yellow",...)
  }
  else if(weighted==TRUE){
    igraph::plot.igraph(graphAdj,layout=layout, rescale=TRUE, 
                        edge.label=round(E(graphAdj)$weight,d=3),
                        vertex.color="yellow",...)
  } 
  else{
    igraph::plot.igraph(graphAdj,layout=layout, rescale=TRUE, 
                        vertex.color="yellow",...)
  }
  
  #### USE mark.groups FOR PLOTTING ELLIPSES AROUND GROUPS OF NODES
  
}
