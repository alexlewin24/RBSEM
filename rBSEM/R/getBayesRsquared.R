#' @title Get posterior summaries of the Bayesian R squared 

#' @description 
#' This function loops over all the multivariate regressions, and for each
#' regression coefficient gets the Bayesian R squared values for each response variable.
#' Currently this function prints to screen the posterior mean and median R squared
#' for each response variable. It also produces (optionally) produces histograms 
#' of the whole posterior R squared for each response. 
#' 
#' The Bayesian R squared is calculated according to Gelman et al. 
#' The American Statistician, Volume 73, 2019.
#' 
#' @param blockGraph adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
#' blockGraph should be a square matrix (adjacency for blocks)
#' columns are LHS of regressions
#' rows are input RHS of regressions
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param outFilePath filepath for RBSEM output files
#' @param outFilePrefix file prefix for RBSEM output files (this will be the name of the input data file, stripped of any .txt)
#' @param varNames vector of variable names from original input data file
#' @param probCI the probability to be contained with the credible intervals

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
#' getBayesRsquared(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'                 outFilePrefix="sem_data",varNames=names(sample_SEM))
#' unlink("tmp", recursive=TRUE)
#' 
#' @return None
#' @export

getBayesRsquared <- function(blockGraph,blockList,outFilePath,outFilePrefix,varNames,
                             probCI=0.95,hist=FALSE){
  
  # whole posterior of the Bayesian R2
      
  nblocks <- dim(blockGraph)[2]
  
  filenumber <- 0
  for( iblock in 1:nblocks ){
    #print(paste("block no.",iblock))

    # 1 or 2 in column of Graph means it is an endogenous variable (so read in R2)
    if( (1 %in% blockGraph[,iblock]) || (2 %in% blockGraph[,iblock]) ){
      
      # get the file for R2 
      # using "full_data" file always: this will use imputed data if necessary 
      filenumber <- filenumber + 1
      fileR2 <- paste(outFilePath,outFilePrefix,"_HESS_R2_full_data_",
                         filenumber,"_MCMC_out.txt",sep="")
      print("Reading from file:")
      print(fileR2)

      # get the dimension of the responses
      qq <- length(varNames[blockList[[iblock]]])
      #print("no. of responses in this block:")
      #print(qq)
      
      # read in the whole posterior R2 file (as 2-dim array)
      R2All <- as.matrix(read.table(fileR2))
      nMCMCiter <- dim(R2All)[1]
      print(paste("no. MCMC iterations = ",nMCMCiter))
            
      # get the response variable names for this block regression
      dimnames(R2All)[[2]] <- varNames[blockList[[iblock]]]
      #print(head(R2All))

      # posterior means and medians
      R2Means <- apply(R2All,FUN=mean,MARGIN=2)
      R2Medians <- apply(R2All,FUN=median,MARGIN=2)
      #print("posterior means:")
      #print(R2Means)
      #print("posterior medians:")
      #print(R2Medians)
      
      # plot histograms of R2
      if(hist) apply(R2All,FUN=hist,MARGIN=2)
     
      # get credible intervals
      R2CredInt <- apply(R2All,MARGIN=2,
            FUN=function(e){ coda::HPDinterval(coda::as.mcmc(e),prob=probCI) })
      #print(R2CredInt)
      
      R2CredInt <- t(R2CredInt)
      R2ResultsTable <- cbind(R2Means,R2Medians,R2CredInt)
      dimnames(R2ResultsTable)[[2]] <- c("mean","median","lower","upper")
      print("Bayesian R-squared summaries for response variables:")
      print(R2ResultsTable)
      
    }
  }
# end function
}



