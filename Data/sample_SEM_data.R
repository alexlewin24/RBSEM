library(mvtnorm)
library(MCMCpack)


## Simulate a 2 block SEM
n = 200
s_1 = 5; s_2 = 3; s=8
p = 20

## Sparsity -- simulate gamma matrices
sparsity_lvl = 0.3
gamma_1 = matrix(NA,p+1,s_1)
for(i in 1:s_1){
  gamma_1[,i] = c(TRUE, (1:p) %in% sample(1:p,p*sparsity_lvl,replace = FALSE) )
}

gamma_2 = matrix(NA,p+s_1+1,s_2)
for(i in 1:(s_2)){
  gamma_2[,i] = c(TRUE, (1:(p+s_1)) %in% sample(1:(p+s_1),(p+s_1)*sparsity_lvl,replace = FALSE) )
}

## Simulate Predictors ...
RX = riwish(p+5, riwish(p+5, diag(p) ) ); RX = solve(diag(sqrt(diag(RX)))) %*% RX %*% solve(diag(sqrt(diag(RX))))  ## rescaled

x = rmvnorm(n,rep(0,p),RX)
x = cbind(rep(1,n),x)

## .. and related Regression Coefficients
sd_b = 3
b_1 = matrix(rnorm((p+1)*s_1,5,sd_b),p+1,s_1)
b_2 = matrix(rnorm((p+s_1+1)*s_2,5,sd_b),p+s_1+1,s_2) ## here we have both beta_2 AND lambdas


## Residual variances and Errors
var_r_1 = diag(rinvgamma(s_1,1.5,2)); var_r_2 = diag(rinvgamma(s_2,1.5,2))
err_1 = matrix(rnorm(n*s_1),n,s_1) %*% chol(var_r_1)
err_2 = matrix(rnorm(n*s_2),n,s_2) %*% chol(var_r_2)

## Finally sample Ys
y=matrix(NA,n,s)
for(i in 1:s_1){
    y[,i] = x[,gamma_1[,i]] %*% b_1[gamma_1[,i],i]
  }
for(i in 1:s_2){
  y[,i+s_1] = cbind(x,y[,1:s_1])[,gamma_2[,i]] %*% b_2[gamma_2[,i],i]
}

y = y + cbind(err_1,err_2)

## Form the data matrix
data = cbind(y,x[,-1])   # leave out the intercept because is coded inside already

## Simulate some missing data
# missing_idx = sample( 1:length(data), 0.05*(n*s) , replace = FALSE ) ##~5% of the data will be deleted
# missing_data = data[missing_idx] ## save them for later checking
# data[missing_idx] = NaN

# Insert the header
block_idx = c(rep(1,s_1),rep(2,s_2),rep(0,p))
### list all the blocks
## starting from x but that's arbitrary

blockL = list( 
  c(9:28),  ## x0 -- block 0
  c(1:5),  ## y1 -- block 1
  c(6:8)  ## y2 -- block 2
)

### then the graph structure
## edge i,j means an arrow i -> j

G = matrix(c( 
  0,1,1,
  0,0,1,
  0,0,0 ), 
  byrow=TRUE,ncol=3,nrow=3)

var_types = rep(0,s_1+s_2+p)

## Write data to file
# write.table(x=data,file="data/sem_data.txt",na = "NAN",col.names=FALSE,row.names=FALSE)
# save.image("data/sample_data.RData")
