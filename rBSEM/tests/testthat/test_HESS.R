context("We're testing the rHESS_SEM function")

test_that("Can read and runs a couple of iterations correctly", {
  dir.create("tmp")
  data(sample_SEM)
  write.table(sample_SEM,"tmp/sem_data.txt",row.names = FALSE,col.names = FALSE)
  expect_equal(
    rHESS_SEM_internal(inFile="tmp/sem_data.txt",outFilePath="tmp/",nIter=20)
    , 0)
  unlink("tmp", recursive=TRUE)
})

test_that("Fails when given non-existing inFile", {
  expect_equal(
    rHESS_SEM_internal(inFile="non_existing.txt",outFilePath="",nIter=20)
    , 1)
})