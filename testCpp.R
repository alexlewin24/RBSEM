cppPrimer = function(inFile, blockList, varType=NULL, SEMGraph, outFilePath="")
{
  
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
      stop("Input file dowsn't exists!")      
    }else{
      
      dataHeader = read.table(inFile,header = FALSE,nrows = 1)
      
      # check the indexes are in the correct range
      if( max(unlist(blockList) ) > length(dataHeader) ){
        stop("blockList indexes provided outside the range in the data matrix!")
      }else{
        if( max(unlist(blockList) ) < length(dataHeader) ){
          blockIndexes = c(blockIndexes,rep(-1, length(dataHeader) - length(blockList) ) )
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
  
  dir.create(outFilePath)
  
  dir.create("tmp/")
  write.table(blockIndexes,"tmp/blockIndexes.txt", row.names = FALSE, col.names = FALSE)
  write.table(varType,"tmp/varType.txt", row.names = FALSE, col.names = FALSE)
  write.table(SEMGraph,"tmp/SEMGraph.txt", row.names = FALSE, col.names = FALSE)
  
}