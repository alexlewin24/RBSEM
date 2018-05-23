#' rBSEM -- Bayesian Structural Equation Models Regression
#' @title traceToArray
#' @description
#' Transform the MCMC trace file resulting from calling rHESS_SEM into a manageable array
#' @name traceToArray
#' @param fileName path to MCMC trace file
#' @param nIterations number of iteration produced by rHESS_SEM
#' @param nPredictors number of predictors expected
#' 
#' @examples 
#'\dontrun{
#'mcmc_gamma_1 = rBSEM::traceToArray(fileName = 
#'   "data/sem_data_HESS_gamma_1_MCMC_out.txt",nIterations = 20000)
#'apply( mcmc_gamma_1 , 1:2 , mean )
#'}
#' @export
traceToArray = function(fileName, nIterations = NULL, nPredictors = NULL){
  
  if( !file.exists( fileName ) )
    stop("Input file doesn't exists!") 
  tmp = as.matrix(  read.table(fileName)  )
    

  if( is.null(nIterations) & is.null(nPredictors) )
    stop("You need to specify at least one between the number of iterations or the number of predictors")
  
  if( is.null(nIterations) )
    nIterations = nrow(tmp) / nPredictors
  
  if( is.null(nPredictors) )
    nPredictors = nrow(tmp) / nIterations
  
  return(
    
    sapply(1:nIterations,  function(i) tmp[(1:nPredictors)+nPredictors*(i-1),] ,simplify = "array")
    
  )
}