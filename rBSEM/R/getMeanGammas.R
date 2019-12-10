#' @title Get the posterior mean gammas (marginal inclusion probabilities)

#' @description 
#' This function does 2 things:
#' prints individual small matrices of gammas for all regressions (gamma.postmean)
#' returns the large gamma matrix for all variables together (gammaAdjgraph)
#' 
#' The output is a square matrix with dimensions m x m where m is the total number 
#' of variables input into the model. Element (j,k) contains the marginal inclusion
#' probability of covariate j in the regression for response variable k. This matrix of
#' posterior probabilities can be thresholded to produce an adjacency matrix in order 
#' to plot the final posterior mean DAG of the model.

#' @param blockGraph adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
#' blockGraph should be a square matrix (adjacency for blocks)
#' columns are LHS of regressions
#' rows are input RHS of regressions
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param outFilePath filepath for RBSEM output files
#' @param outFilePrefix file prefix for RBSEM output files (this will be the name of the input data file, stripped of any .txt)
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
#' rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
#'          SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0,nChains = 2)
#' getMeanGammas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'          outFilePrefix="sem_data",varNames=names(sample_SEM))
#' unlink("tmp", recursive=TRUE)
#'  
#' 
#' @return Matrix of posterior mean gammas (marginal inclusion probabilities) 
#' in adjacency matrix format. Columns are response variables, rows are
#' covariates in regression models. 
#' 
#' @export

getMeanGammas <- function(blockGraph,blockList,outFilePath,outFilePrefix,varNames){

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
  
  nblocks <- dim(blockGraph)[2]
  
  # get total number of variables used, now done above
  #ntot <- length(unlist(blockList))
  
  # initialise adjacency matrix for all variables 
  gammaAdjgraph <- matrix(0,nrow=ntot,ncol=ntot)
  #print(adjgraph)
  # initialise row and column names of expanded gamma matrix
  columnNames <- rep("",ntot)
  
  filenumber <- 0
  for( iblock in 1:nblocks ){
    print(paste("block no.",iblock))
    columnNames[blockList[[iblock]]] <- varNames[blockList[[iblock]]]
    
    # 1 in column of Graph means it is an endogenous variable (so read in gammas)
    if( 1 %in% blockGraph[,iblock] ){
      
      # get the posterior mean gammas 
      filenumber <- filenumber + 1
      filegamma <- paste(outFilePath,outFilePrefix,"_HESS_gamma_",
                         filenumber,"_out.txt",sep="")
      print(filegamma)
      gamma.postmean <- as.matrix(read.table(filegamma))
      
      # get the column variable names (response variables) for this block regression
      colnames(gamma.postmean) <- varNames[blockList[[iblock]]]
      
      # get the row variable names (predictor variables, may be from more than one block)
      dummy <- NULL
      dumindex <- NULL
      for( jblock in which(blockGraph[,iblock]==1) ){
        print(jblock)
        dummy <- c(dummy,varNames[blockList[[jblock]]])
        print(dummy)
        dumindex <- c(dumindex,blockList[[jblock]])
        print(dumindex)
      }
      rownames(gamma.postmean) <- dummy
      print("posterior means gammas:")
      print("columns are response variables, rows are covariates")
      print(gamma.postmean)
      
      p <- dim(gamma.postmean)[1]
      q <- dim(gamma.postmean)[2]
      #print(p);print(q)
      
      gammaAdjgraph[dumindex,blockList[[iblock]]] <- gamma.postmean
        
    }
  }
  colnames(gammaAdjgraph) <- columnNames
  rownames(gammaAdjgraph) <- columnNames
  #print(gammaAdjgraph)
  
  return(gammaAdjgraph)
  
# end function
}



