#' @title Get posterior credible intervals for regression coefficients 
#' (conditional on variable selected into model)

#' @description 
#' This function loops over all the multivariate regressions, and for each
#' regression coefficient calculates a Highest Posterior Density credible
#' interval (using the \code{\link{coda}} package). In this function these 
#' are calculated conditional on the variable being selected in the model. 
#' Use \code{\link{getCredIntBetasUnconditional}} for unconditional 
#' posterior summaries.
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

########### small function used to get the credible intervals
funHPD <- function(e,probCI=probCI){
  if(length(e)>1){
    res <- coda::HPDinterval(coda::as.mcmc(e),prob=probCI)
  }
  else{
    res <- array(c(NA,NA),dim=c(1,2),dimnames=list("",c("lower","upper")))
  }
}

#############

getCredIntBetasConditional <- function(blockGraph,blockList,outFilePath,outFilePrefix,varNames,
                            Intercept=TRUE,probCI){
  
  # credible intervals from whole posterior (think conditional or shrunk, easier shrunk)
    
  # blockGraph adjacency matrix for blocks
  # blockList list of variables in each block
  # outFilePath filepath for RBSEM output files
  # varNames vector of variable names from original input data file
  
  # iblock are the responses, jblock are the predictors
  
  nblocks <- dim(blockGraph)[2]
  
  betaFinalList <- vector("list",nblocks)
  
  filenumber <- 0
  for( iblock in 1:nblocks ){
    print(paste("block no.",iblock))

    # 1 in column of Graph means it is an endogenous variable with VS (so read in betas)
    if( (1 %in% blockGraph[,iblock]) ){
      
      # get the whole posterior of betas and gammas
      filenumber <- filenumber + 1
      filebeta <- paste(outFilePath,outFilePrefix,"_HESS_beta_",
                         filenumber,"_MCMC_out.txt",sep="")
      filegamma <- paste(outFilePath,outFilePrefix,"_HESS_gamma_",
                        filenumber,"_MCMC_out.txt",sep="")
      print("Reading from files:")
      print(filebeta)
      print(filegamma)
      
      # get the row variable names (predictor variables, may be from more than one block)
      # keepindex indicates the variables also have gammas
      if(Intercept){
        predictorNames <- "intercept"
        dumindex <- NA
        keepindex <- FALSE
      } else{
        predictorNames <- NULL
        dumindex <- NULL
        keepindex <- NULL
      }
      for( jblock in which(blockGraph[,iblock]>0)  ){
        #print(jblock)
        predictorNames <- c(predictorNames,varNames[blockList[[jblock]]])
        dumindex <- c(dumindex,blockList[[jblock]])
        if( blockGraph[jblock,iblock]==1 ) 
          keepindex <- c(keepindex,rep(TRUE,length(blockList[[jblock]])))
        else keepindex <- c(keepindex,rep(FALSE,length(blockList[[jblock]])))
        #print(predictorNames)
        #print(dumindex)
        #print(keepindex)
      }
      
      # get the dimension of the responses
      qq <- length(varNames[blockList[[iblock]]])
      # get the dimension of the covariates
      pp <- length(predictorNames)
      #print("dimensions of beta matrix including forced parameters:")
      #print(pp);print(qq)
      
      # read in the whole posterior beta file (as 2-dim array)
      betaAll <- as.matrix(read.table(filebeta))
      nMCMCiter <- dim(betaAll)[1]/pp
      print(paste("no. MCMC iterations for betas = ",nMCMCiter))
      
      # reshape whole posterior as 3-dim array 
      betaPosteriors <- array(betaAll,dim=c(pp,nMCMCiter,qq))
      
      # get the response variable names for this block regression
      dimnames(betaPosteriors) <- vector("list", 3)
      dimnames(betaPosteriors)[[3]] <- varNames[blockList[[iblock]]]
      dimnames(betaPosteriors)[[1]] <- predictorNames
    
      ## unconditional posterior summaries
      # posterior means
      betaMeans <- apply(betaPosteriors,FUN=mean,MARGIN=c(1,3))
      # posterior medians
      betaMedians <- apply(betaPosteriors,FUN=median,MARGIN=c(1,3))
            # get credible intervals
      betaCredInt <- apply(betaPosteriors,MARGIN=c(1,3),
            FUN=function(e){ coda::HPDinterval(coda::as.mcmc(e),prob=probCI) })
      dimnames(betaCredInt)[[1]] <- c("lower","upper")
      betaCredInt <- aperm(betaCredInt,c(2,1,3))

      betaResultsTable <- abind(betaMeans,betaMedians,betaCredInt,along=2)
      dimnames(betaResultsTable)[[2]] <- c("mean","median","lower","upper")
      # print("Posterior summaries for regression coefs:")
      # print("One table for each response variable")
      # print("Here means are unconditional. 0 if not selected.")
      # print(betaResultsTable)
      
      ## conditional summaries
      
      ## first remove betas which are not subject to VS
      betaPosteriorsKeep <- betaPosteriors[keepindex,,]
      pp_cond <- dim(betaPosteriorsKeep)[1]
      dimnames1 <- dimnames(betaPosteriors)
      dimnames1[[1]] <- dimnames(betaPosteriors)[[1]][keepindex]
      betaPosteriorsKeep <- array(betaPosteriorsKeep,dim=c(pp_cond,nMCMCiter,qq),
                                  dimnames=dimnames1)
      #print("dim of all beta posteriors")
      #print(dim(betaPosteriors))
      #print("dim of beta posteriors subject to VS")
      #print(dim(betaPosteriorsKeep))
      print("names in betaPosteriorsKeep")
      print(dimnames(betaPosteriorsKeep))
      print("no. of beta variables subjecct to VS:")
      print(pp_cond)
      
      # read in the whole posterior gamma file (as 2-dim array)
      gammaAll <- as.matrix(read.table(filegamma))
      nMCMCiter <- dim(gammaAll)[1]/pp_cond
      print(paste("no. MCMC iterations for gammas = ",nMCMCiter))
      
      # reshape whole posterior as 3-dim array 
      gammaPosteriors <- array(gammaAll,dim=c(pp_cond,nMCMCiter,qq))
      
      # get the response variable names for this block regression
      dimnames(gammaPosteriors) <- vector("list", 3)
      dimnames(gammaPosteriors)[[3]] <- varNames[blockList[[iblock]]]
      dimnames(gammaPosteriors)[[1]] <- predictorNames[keepindex]
      
      # calc mean gammas (to check)
      gammaMeans <- apply(gammaPosteriors,FUN=mean,MARGIN=c(1,3))
      gammaMeans <- array(gammaMeans,dim=c(pp_cond,1,qq))
      dimnames(gammaMeans) <- vector("list",3)
      dimnames(gammaMeans)[[1]] <- dimnames(gammaPosteriors)[[1]]
      dimnames(gammaMeans)[[3]] <- dimnames(gammaPosteriors)[[3]]
      # print("mean gammas:")
      # print(gammaMeans)
      
      ####### calc beta posterior summaries conditional on gamma=1
      
      # convert gamma 1/0 to T/F  
      gammaPosteriorsBool <- array(as.logical(gammaPosteriors),dim(gammaPosteriors))

      # convert beta and gamma posterior arrays to lists
      betaPosteriorsKeepList <- split(betaPosteriorsKeep,slice.index(betaPosteriorsKeep,MARGIN = c(1,3)))
      gammaPosteriorsBoolList <- split(gammaPosteriorsBool,slice.index(gammaPosteriorsBool,MARGIN = c(1,3)))
      
      # this is a list, each element contains conditional posterior for a single beta
      betaPosteriorsCond <- mapply(function(a,b){a[b]},betaPosteriorsKeepList,gammaPosteriorsBoolList)
      #print("lengths of posterior conditional betas:")
      #print(lengths(betaPosteriorsCond))
            
      # get the posterior summaries
      # sapply gives results in vector/matrix format
      betaCondMeans <- sapply(betaPosteriorsCond,FUN=mean)
      betaCondMedians <- sapply(betaPosteriorsCond,FUN=median)
      betaCondCIs <- sapply(betaPosteriorsCond,FUN=funHPD,probCI=probCI)
  
      betaCondResultsTable <- rbind(betaCondMeans,betaCondMedians,betaCondCIs)
      dimnames(betaCondResultsTable)[[1]] <- c("mean","median","lower","upper")
      betaCondResultsArray <- array(betaCondResultsTable,dim=c(4,pp_cond,qq))
      betaCondResultsArray <- aperm(betaCondResultsArray,c(2,1,3))
      dimnames2 <- dimnames(betaPosteriorsKeep)
      dimnames2[[2]] <- c("cond mean","cond median","cond lower","cond upper")
      dimnames(betaCondResultsArray) <- dimnames2
      
      # print("Posterior summaries for regression coefs:")
      # print("One table for each response variable")
      # print("Here means ARE conditional on being selected. NA or 0 if not selected.")
      # print(betaCondResultsArray)
      
      # put uncond and cond together 
      #print(dim(betaResultsTable))
      betaUncondArray <- array(betaResultsTable[keepindex,,],dim=c(pp_cond,4,qq))
      #print(dim(betaUncondArray))
      #print(dim(betaCondResultsArray))
      betaFinalTable <- abind(gammaMeans,betaUncondArray,betaCondResultsArray,along=2)
      dimnames(betaFinalTable)[[2]] <- c("MPPI","beta mean","median","lower","upper",
                         "cond mean","cond median","cond lower","cond upper")
      print(betaFinalTable)
      
      betaFinalList[[iblock]] <- betaFinalTable
      
    }
  }

  print("IMPORTANT NOTE:")
  print("Credible intervals (highest posterior intervals) are 
        not useful summaries of uncertainty for regression parameters with 
        low posterior probability of association. 
        Values above should only be used for regresssion
        parameters selected to be in the model.")
  
  print("This function returns a list of 3D arrays. Each element in the
        list corresponds to 1 multivariate regression. Posterior results
        for 1 regression is a 3D array, 1st dimension predictors,
        2nd dimension posterior summaries, 3rd dimension response variables.")

  return(betaFinalList)
# end function
}



