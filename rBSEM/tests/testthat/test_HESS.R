context("We're testing the rHESS_SEM function")

test_that("Fails when given non-existing inFile", {
  
  blockL = list( 
    c(9:28),  ## x0 -- block 0
    c(1:5),  ## y1 -- block 1
    c(6:8)  ## y2 -- block 2
  )
  G = matrix(c( 
    0,1,1,
    0,0,1,
    0,0,0 ), 
    byrow=TRUE,ncol=3,nrow=3)
  
  
  expect_that(
    rHESS_SEM(inFile="non_existing.txt",outFilePath="",blockList = blockL,SEMGraph = G,nIter=50)
    , throws_error())
})

test_that("Can read and runs a couple of iterations correctly with method 0 and 1 chain", {
  
  dir.create("tmp")
  data(sample_SEM)
  write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
  blockL = list( 
    c(9:28),  ## x0 -- block 0
    c(1:5),  ## y1 -- block 1
    c(6:8)  ## y2 -- block 2
  )
  G = matrix(c( 
    0,1,1,
    0,0,1,
    0,0,0 ), 
    byrow=TRUE,ncol=3,nrow=3)

  expect_equal(
    rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
                     SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0)
    , 0)
  unlink("tmp", recursive=TRUE)
})

test_that("Can read and runs a couple of iterations correctly with method 0 and 2 chains", {
  
  dir.create("tmp")
  data(sample_SEM)
  write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
  blockL = list( 
    c(9:28),  ## x0 -- block 0
    c(1:5),  ## y1 -- block 1
    c(6:8)  ## y2 -- block 2
  )
  G = matrix(c( 
    0,1,1,
    0,0,1,
    0,0,0 ), 
    byrow=TRUE,ncol=3,nrow=3)
  
  expect_equal(
    rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
                     SEMGraph = G,outFilePath="tmp/",nIter=50,method = 0,nChains = 2)
    , 0)
  unlink("tmp", recursive=TRUE)
})

test_that("Can read and runs a couple of iterations correctly with method 1 and 1 chain", {
  
  dir.create("tmp")
  data(sample_SEM)
  write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
  blockL = list( 
    c(9:28),  ## x0 -- block 0
    c(1:5),  ## y1 -- block 1
    c(6:8)  ## y2 -- block 2
  )
  G = matrix(c( 
    0,1,1,
    0,0,1,
    0,0,0 ), 
    byrow=TRUE,ncol=3,nrow=3)
  
  expect_equal(
    rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
                     SEMGraph = G,outFilePath="tmp/",nIter=50,method = 1)
    , 0)
  unlink("tmp", recursive=TRUE)
})

test_that("Can read and runs a couple of iterations correctly with method 1 and 2 chains", {
  
  dir.create("tmp")
  data(sample_SEM)
  write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
  blockL = list( 
    c(9:28),  ## x0 -- block 0
    c(1:5),  ## y1 -- block 1
    c(6:8)  ## y2 -- block 2
  )
  G = matrix(c( 
    0,1,1,
    0,0,1,
    0,0,0 ), 
    byrow=TRUE,ncol=3,nrow=3)
  
  expect_equal(
    rBSEM::rHESS_SEM(inFile="tmp/sem_data.txt",blockList = blockL,
                     SEMGraph = G,outFilePath="tmp/",nIter=50,method = 1, nChains = 2)
    , 0)
  unlink("tmp", recursive=TRUE)
})