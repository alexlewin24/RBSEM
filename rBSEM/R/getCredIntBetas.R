#' @title Get posterior credible intervals for regression coefficients 

#' @description 
#' This function loops over all the multivariate regressions, and for each
#' regression coefficient calculates a Highest Posterior Density credible
#' interval (using the \code{\link{coda}} package).
#' 
#' @param blockGraph adjacency matrix for blocks (as input to \code{\link{rHESS_SEM}})
#' blockGraph should be a square matrix (adjacency for blocks)
#' columns are LHS of regressions
#' rows are input RHS of regressions
#' @param blockList list of variables in each block (as input to \code{\link{rHESS_SEM}})
#' @param outFilePath filepath for RBSEM output files
#' @param outFilePrefix file prefix for RBSEM output files (this will be the name of the input data file, stripped of any .txt)
#' @param varNames vector of variable names from original input data file
#' @param Intercept does the model include intercept terms? (default TRUE)
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
#' getMeanBetas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'          outFilePrefix="sem_data",varNames=names(sample_SEM))
#' getCredIntBetas(blockList = blockL,blockGraph = G,outFilePath="tmp/",
#'          outFilePrefix="sem_data",varNames=names(sample_SEM),probCI=0.9)
#' unlink("tmp", recursive=TRUE)
#' 
#' @return None
#' @export
#' @importFrom coda as.mcmc HPDinterval

getCredIntBetas <- function(blockGraph,blockList,outFilePath,outFilePrefix,varNames,
                            Intercept=TRUE,probCI){
  
  # credible intervals from whole posterior (think conditional or shrunk, easier shrunk)
    
  # blockGraph adjacency matrix for blocks
  # blockList list of variables in each block
  # outFilePath filepath for RBSEM output files
  # varNames vector of variable names from original input data file
  
  nblocks <- dim(blockGraph)[2]
  
  filenumber <- 0
  for( iblock in 1:nblocks ){
    print(paste("block no.",iblock))

    # 1 or 2 in column of Graph means it is an endogenous variable (so read in betas)
    if( (1 %in% blockGraph[,iblock]) || (2 %in% blockGraph[,iblock]) ){
      
      # get the whole posterior of betas 
      filenumber <- filenumber + 1
      filebeta <- paste(outFilePath,outFilePrefix,"_HESS_beta_",
                         filenumber,"_MCMC_out.txt",sep="")
      print(filebeta)
      
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
      
      # get the dimension of the responses
      qq <- length(varNames[blockList[[iblock]]])
      # get the dimension of the covariates
      pp <- length(dummy)
      print("dimensions of beta matrix:")
      print(pp);print(qq)
      
      # read in the whole posterior beta file (as 2-dim array)
      betaAll <- as.matrix(read.table(filebeta))
      #print(dim(betaAll))
      #print(head(betaAll))
      nMCMCiter <- dim(betaAll)[1]/pp
      print(paste("no. MCMC iterations = ",nMCMCiter))
      
      # reshape whole posterior as 3-dim array 
      betaPosteriors <- array(betaAll,dim=c(pp,nMCMCiter,qq))
      
      # get the response variable names for this block regression
      dimnames(betaPosteriors) <- vector("list", 3)
      dimnames(betaPosteriors)[[3]] <- varNames[blockList[[iblock]]]
      dimnames(betaPosteriors)[[1]] <- dummy
      
      # print(betaPosteriors)
      # print("element [1,1] of matrix, whole posterior:")
      # print(betaPosteriors[1,,1])
      # print("element [1,2] of matrix, whole posterior:")
      # print(betaPosteriors[1,,2])
      # print("element [2,1] of matrix, whole posterior:")
      # print(betaPosteriors[2,,1])
      # print("element [2,2] of matrix, whole posterior:")
      # print(betaPosteriors[2,,2])
      
      # check posterior means
      betaMeans <- apply(betaPosteriors,FUN=mean,MARGIN=c(1,3))
      print("shrunk posterior means for checking:")
      print(betaMeans)
      
      # get credible intervals
      # test first
      #hist(betaPosteriors[1,,1])
      #testmcmc <- coda::as.mcmc(betaPosteriors[1,,1])
      #print(coda::HPDinterval(testmcmc, prob = 0.95))
      
      betaCredInt <- apply(betaPosteriors,MARGIN=c(1,3),
            FUN=function(e){ coda::HPDinterval(coda::as.mcmc(e),prob=probCI) })
      dimnames(betaCredInt)[[1]] <- c("lower","upper")
      #print(dim(betaCredInt))
      print(paste("posterior ",probCI," credible intervals"))
      print(betaCredInt)
    }
  }

# end function
}



