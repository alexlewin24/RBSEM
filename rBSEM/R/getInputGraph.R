#' @title Creates the maximal adjacency matrix allowed for the variables input into the model. 

#' @description 
#' The rBSEM package paramterises the Bayesian SEM model through a DAG
#' based on the blocks, plus a list of variables for each block.
#' 
#' This function expands the input DAG between blocks
#' of variables to give an adjacency matrix between variables. 
#' It is useful if you want to plot a DAG showing which input variables you
#' are allowing into the model (before fitting the model).

#' @param blockGraph adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
#' blockGraph should be a square matrix (adjacency for blocks)
#' columns are LHS of regressions
#' rows are input RHS of regressions
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param varNames vector of variable names from original input data file
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
#' getInputGraph(blockList = blockL,blockGraph = G,varNames=names(sample_SEM))
#' unlink("tmp", recursive=TRUE)
#'  
#' @return Adjacency Matrix of input model
#' 
#' @export
#' 


getInputGraph <- function(blockGraph,blockList,varNames=NULL){
  
  # blockGraph adjacency matrix for blocks
  # blockList list of variables in each block

  # blockGraph should be a square matrix (adjacency for blocks)
  # columns are LHS of regressions
  # rows are input RHS of regressions
  nblocks <- dim(blockGraph)[2]
  print(paste("no. blocks = ",nblocks))
  
  # here redefine get block list, var names just the ones we want 
  # so adjgraph indices not out of bounds
  # variable names: just the ones we need, in order of blocks
  varNames <- varNames[unlist(blockList)]
  print("variable names:")
  print(varNames)
  # block list: indices in order, just maintain correct block sizes
  ntot <- length(varNames)
  blockList <- relist(flesh=1:ntot,skeleton=blockList)
  print("re-ordered block List:")
  print(blockList)
  
  # initialise adjacency matrix for all variables 
  adjgraph <- matrix(0,nrow=ntot,ncol=ntot)
  #print(adjgraph)
  
  # initialise row and column names (if varNames given)
  if(!is.null(varNames)){
    columnNames <- rep("",ntot)
  } 
  
  for( iblock in 1:nblocks ){
    
    if(!is.null(varNames)) columnNames[blockList[[iblock]]] <- varNames[blockList[[iblock]]]
    
    # 1 in column of Graph means it is an endogenous variable
    if( 1 %in% blockGraph[,iblock] ){
      print(paste("block no.",iblock))
      #print(blockList[[iblock]])
      
      # get no. Y variables in this block
      numberColumns <- length(blockList[[iblock]])
      print(paste("no. cols in block = ",numberColumns))
  
      # get no. X variables in this block (may be from more than one block)
      for( jblock in which(blockGraph[,iblock]==1) ){
        #print(jblock)
        numberRows <- length(blockList[[jblock]])
        print(paste("no. rows in block = ",numberRows))
        adjgraph[blockList[[jblock]],blockList[[iblock]]] <- 1
        #print(adjgraph)
      }
    }
  }

  if(!is.null(varNames)){
    colnames(adjgraph) <- columnNames
    rownames(adjgraph) <- columnNames
  }
  #print("final graph")
  #print(adjgraph)
  
  return(adjgraph)
# end function      
}

