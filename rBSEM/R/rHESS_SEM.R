#' rBSEM -- Bayesian Structural Equation Models Regression
#' @title rHESS_SEM
#' @description
#' Run a simple SEM Bayesian sampler
#' @name rHESS_SEM
#' @param inFile path to data file; plain text file with variables on the columns and observations on the rows 
#' @param blockList list of blocks in the model; each element of the list contains the (column) indices of the variables in each block, with respect to the data file
#' @param varType variable type for each column in the data file; coded as: 0 - continuous, 1- binary, 2 - categorical. Note that categorical variables cannot be imputed
#' @param SEMGraph graph adjacency matrix representing the SEM structure between the blocks. Edges represented as 2 indicate variables that will be always included in the regression model
#' @param outFilePath path to where the output files are to be written
#' @param nIter number of iterations for the MCMC procedure
#' @param burnin number of iterations (or fraction of iterations) to discard at the start of the chain
#' @param nChains number of parallel chains to run
#' @param autoAddIntercept should the c++ code automatically add an intercept to every equation? Default: TRUE
#' @param gammaInit gamma initialisation to either all-zeros ("0"), all ones ("1"), randomly ("R") or (default) MLE-informed ("MLE").
#' @param seed pRNG seed
#' @param method \deqn{\gamma}{gamma} sampling method, where 0=\deqn{MC^3}{MC^3} and 1=Thompson-sampling inspired novel method
#' @param writeOutputLevel 1=no imputed values output to file, 2=imputed values are output to file (future implement 0 for only posterior means output)
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
#'   rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
#'                    SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0,nChains = 2)
#' unlink("tmp", recursive=TRUE)
#' 
#' @export
rHESS_SEM = function(inFile, blockList, varType=NULL, SEMGraph, outFilePath="", autoAddIntercept=TRUE, gammaInit="S" ,nIter, burnin=0, nChains=1, seed=0, method=1, writeOutputLevel=1)
{
  
  # cleanup file PATHS
  inFileLength = nchar(inFile)
  if( inFileLength == 0 )
    stop("Please provide a correct path to a plain-text (.txt) file")
  
  if( substr(inFile,1,1) == "~" )
    inFile = path.expand(inFile)
  
  outFilePathLength = nchar(outFilePath)
  if( outFilePathLength > 0 )
  {
    if( substr(outFilePath,outFilePathLength,outFilePathLength) != "/" )
      paste( outFilePath , "/" , sep="" )
  }
  
  # blockList
  if( length(blockList) < 2 ){
    
    stop("Need at least 2 blocks!")
  
  }else{
    
    blockIndexes = rep(NA, max(unlist(blockList)))
    for( i in 1:length(blockList))
      blockIndexes[blockList[[i]]] = i-1
    
    blockIndexes[is.na(blockIndexes)] = -1
    
    # now try and read from given inFile
    if( !file.exists( inFile ) ){
      stop("Input file doesn't exists!")      
    }else{
      
      dataHeader = read.table(inFile,header = FALSE,nrows = 1)
      
      # check the indexes are in the correct range
      if( max(unlist(blockList) ) > length(dataHeader) ){
        stop("blockList indexes provided outside the range in the data matrix!")
      }else{
        if( max(unlist(blockList) ) < length(dataHeader) ){
          blockIndexes = c(blockIndexes,rep(-1, length(dataHeader) - length(blockIndexes) ) )
        }
        # else is fine, it means they're equal
      }
      
    }
  }
  
  # default varType
  if(is.null(varType)){
    varType = rep( 0,length(dataHeader) )
  }else{
    
    if( length(varType) > length(dataHeader) ){
      stop("more varible types provided than columns in data matrix")
    }else{
      if( length(varType) < length(dataHeader) ){
        
        if( length(varType) != sum(blockIndexes!=-1) ){
          stop("less varible types provided than used columns in data matrix")
        }
        
      }# else is fine
    }
  }
  
  ## graph
  if( is.null(dim(SEMGraph)) | !is.matrix(SEMGraph) ){
    stop("The graph structure must be given as an adjacency matrix")
  }else{
    if( ncol(SEMGraph) != length(blockList) ){
      stop("The graph structure must have the same number of blocks as blockList")
    }else{
     
      ## check no loops and at least one covariate-only group
      if( any( ((SEMGraph!=0) + (t(SEMGraph)!=0))>1 )  )
        stop("The graph should contain no loops nor diagonal elements")
      
      if( !any( colSums(SEMGraph) == 0 ) )
        stop("At least one set of varaible needs to be covariates (right-hand-side) only")
      
      ## else is fine       
    }
  }


  # check how burnin was given
  if ( burnin < 0 ){
    stop("Burnin must be positive or 0")
  }else{ if ( burnin > nIter ){
    stop("Burnin might ont be greater then nIter")
  }else{ if ( burnin < 1 ){ # given as a fraction
    burnin = ceiling(nIter * burnin) # the zero case is taken into account here as well
  }}} # else assume is given as an absolute number
  
  dir.create(outFilePath)
  
  dir.create("tmp/")
  write.table(blockIndexes,"tmp/blockIndexes.txt", row.names = FALSE, col.names = FALSE)
  write.table(varType,"tmp/varType.txt", row.names = FALSE, col.names = FALSE)
  write.table(SEMGraph,"tmp/SEMGraph.txt", row.names = FALSE, col.names = FALSE)
  
  status = rHESS_SEM_internal(inFile, outFilePath, autoAddIntercept, gammaInit, nIter, burnin, nChains, seed, method, writeOutputLevel)

  if(outFilePath != "tmp/")
    unlink("tmp",recursive = TRUE)
  
  return(status)
}