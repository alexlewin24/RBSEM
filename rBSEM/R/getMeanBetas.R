#' @title Get the conditional posterior mean regression coefficients 

#' @description 
#' This function does 2 things:
#' prints individual small matrices of betas for all regressions (beta.postmean)
#' returns the large beta matrix for all variables together (betaAdjgraph)
#' 
#' The output is a square matrix with dimensions m x m where m is the total number 
#' of variables input into the model. Element (j,k) contains the posterior mean 
#' of the regression coefficient for covariate j in the regression for 
#' response variable k, conditional on that covariate being selected into the
#' regression. The unconditional (shrunk) coefficients can be found by multiplying
#' by the posterior inclusion probability (see \code{\link{getMeanGammas}}).
#' 
#' There may also be covariates which are forced into some regressions: these
#' will have betas but not gammas. 
#' 
#' The output beta matrix does not include the intercept terms from the regression.
#' However these are printed as part of the small matrices for the individual 
#' regressions. 

#' @param blockGraph adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
#' blockGraph should be a square matrix (adjacency for blocks)
#' columns are LHS of regressions
#' rows are input RHS of regressions
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param outFilePath filepath for RBSEM output files
#' @param outFilePrefix file prefix for RBSEM output files (this will be the name of the input data file, stripped of any .txt)
#' @param varNames vector of variable names from original input data file
#' @param Intercept does the model include intercept terms? (default TRUE)
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
#' getMeanBetas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'          outFilePrefix="sem_data",varNames=names(sample_SEM))
#' unlink("tmp", recursive=TRUE)
#'  
#' @return Matrix of posterior mean regression coefficients 
#' (conditional on being selected in the model) in adjacency matrix format.
#' Columns are response variables, rows are covariates. 
#' 
#' @export

getMeanBetas <- function(blockGraph,blockList,outFilePath,outFilePrefix,varNames,
                         Intercept=TRUE){
  
  if(Intercept) print("Here assuming model INCLUDES INTERCEPTS; change Intercept argument if incorrect")
  else print("Here assuming model had NO INTERCEPTS; change Intercept argument if incorrect")
  
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
  
  # initialise adjacency matrix for all variables 
  betaAdjgraph <- matrix(0,nrow=ntot,ncol=ntot)
  #print(adjgraph)
  # initialise row and column names of expanded beta matrix
  columnNames <- rep("",ntot)
  
  filenumber <- 0
  for( iblock in 1:nblocks ){
    print(paste("block no.",iblock))
    columnNames[blockList[[iblock]]] <- varNames[blockList[[iblock]]]
    
    # 1 or 2 in column of Graph means it is an endogenous variable (so read in betas)
    if( (1 %in% blockGraph[,iblock]) || (2 %in% blockGraph[,iblock]) ){
      
      # get the posterior mean betas 
      filenumber <- filenumber + 1
      filebeta <- paste(outFilePath,outFilePrefix,"_HESS_beta_",
                         filenumber,"_out.txt",sep="")
      print(filebeta)
      beta.postmean <- as.matrix(read.table(filebeta))
      
      # get the column variable names (response variables) for this block regression
      colnames(beta.postmean) <- varNames[blockList[[iblock]]]
      
      # get the row variable names (predictor variables, may be from more than one block)
      if(Intercept){
        dummy <- "intercept"
        dumindex <- NA
      } else{
        dummy <- NULL
        dumindex <- NULL
      }
      for( jblock in which(blockGraph[,iblock]>0)  ){
        print(jblock)
        dummy <- c(dummy,varNames[blockList[[jblock]]])
        print(dummy)
        dumindex <- c(dumindex,blockList[[jblock]])
        print(dumindex)
      }
      print(paste("expecting no. coefs = ",length(dummy)))
      print(paste("no. coefs in output file = ",dim(beta.postmean)[1]))
      rownames(beta.postmean) <- dummy
      print("posterior mean regression coefficients, conditional on being selected:")
      print("columns are response variables, rows are covariates")
      print(beta.postmean)
      
      p <- dim(beta.postmean)[1]
      q <- dim(beta.postmean)[2]
      #print(p);print(q)
      
      # put the small beta matrix into a block of the big matrix
      # if intercepts, remove first index and first row here (not needed in DAG)
      if(Intercept) betaAdjgraph[dumindex[-1],blockList[[iblock]]] <- beta.postmean[-1,]
      else betaAdjgraph[dumindex,blockList[[iblock]]] <- beta.postmean
    }
  }
  colnames(betaAdjgraph) <- columnNames
  rownames(betaAdjgraph) <- columnNames
  #print(betaAdjgraph)
  
  return(betaAdjgraph)
  
# end function
}



