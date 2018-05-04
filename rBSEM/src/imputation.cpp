#include "imputation.h"

namespace Imputation{

    void initialiseXimpute(arma::mat& data, arma::uvec& missingDataIndexes, arma::umat& SEMGraph, std::vector<arma::uvec>& blockIdx,
                    arma::vec& covariatesOnlyMean, arma::vec& covariatesOnlyVar, arma::uvec& covariatesOnlyIdx) // return values 
    {

        unsigned int nBlocks = SEMGraph.n_cols;
        unsigned int nObservations = data.n_rows;

        arma::uvec blockCovariatesOnlyIdx(nBlocks); // block labelss of variables that appear only on the rhs
        covariatesOnlyIdx = arma::uvec( data.n_cols ); // indexes of variables that appear only on the rhs

        unsigned int blockCovLeft = 0;
        unsigned int covLeft = 0;

        arma::uvec tmpToAdd;

        for( unsigned int k=0; k<nBlocks; ++k)
        {
            if( arma::all( SEMGraph.col(k) == 0 ) )
            {
                // this block is made of rhs-only variables
                blockCovariatesOnlyIdx(blockCovLeft++) = k;
                // add the variables indexes
                tmpToAdd = blockIdx[k];
                covariatesOnlyIdx.subvec( covLeft, covLeft + tmpToAdd.n_elem -1 ) = tmpToAdd;
                covLeft += tmpToAdd.n_elem;

            }else{

                // this block is made of lhs variables (for some equations)
                // so no need to do anything, the model structure will tell us how to init afterward..

            }
        }

        blockCovariatesOnlyIdx.resize(blockCovLeft); // resize it to correct dimension
        covariatesOnlyIdx.resize(covLeft); // resize it to correct dimension


        // now on principle we'd like to impute this covariatesOnly via a (latent) MvNormal of sort. TODO
        // for now impute them using independent univariates normals
        covariatesOnlyMean = arma::vec(covariatesOnlyIdx.n_elem);
        covariatesOnlyVar = arma::vec(covariatesOnlyIdx.n_elem);

        double mean = 0.0, M2=0.0, delta=0.0, delta2=0.0;
        unsigned int count = 0;

        // compute mean and variance of the imputation distribution
        for( unsigned int j=0, n=covariatesOnlyIdx.n_elem; j<n; ++j)
        {
            for( unsigned int i=0; i<nObservations; ++i )
            {
                if( (data(i,covariatesOnlyIdx(j)) == data(i,covariatesOnlyIdx(j))) ) // definition of a non-NaN
                {
                    count++;
                    delta = data(i,covariatesOnlyIdx(j)) - mean;
                    mean = mean + delta / (double)count;
                    delta2 = data(i,covariatesOnlyIdx(j)) - mean;
                    M2 = M2 + delta * delta2;
                }
            }

            covariatesOnlyMean(j) = mean;
            covariatesOnlyVar(j) = M2/(double)(count-1);
        }
        
    }


    void imputeAll(arma::mat& data, const arma::uvec& missingDataIndexes, const arma::umat& missingDataIdxArray, arma::ivec& varType,
                    const arma::vec& covariatesOnlyMean, const arma::vec& covariatesOnlyVar, const arma::uvec& covariatesOnlyIdx,
                    const std::vector<arma::uvec>& outcomesIdx, const std::vector<arma::uvec>& fixedPredictorsIdx, 
                    const std::vector<arma::uvec>& vsPredictorsIdx,
                    const std::vector<arma::ucube>& gamma_state, const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0 )
    {

        unsigned int nObservations = data.n_rows;

        if( missingDataIdxArray.n_elem > 0 ){


            // actually impute the values
            arma::vec tmpVec;
            arma::uvec missingIdxThisColumn;
            arma::uvec currentCol(1);
            unsigned int nMissingThisColumn;
            double aCovariates, bCovariates;
            double mCovariates, vSquareCovariates;
            double V_n;
            
            for( unsigned int j=0, n=covariatesOnlyIdx.n_elem; j<n; ++j)
            {
                currentCol = covariatesOnlyIdx(j);
                // check howManyNaNInThisColumn
                missingIdxThisColumn = missingDataIdxArray.submat( arma::find( missingDataIdxArray.col(1) == covariatesOnlyIdx(j) ) , arma::zeros<arma::uvec>(1) ); // arma::zeros<arma::uvec>(1) is basically one '0', but in uvec shape
                nMissingThisColumn = missingIdxThisColumn.n_elem;
                if( nMissingThisColumn > 0 )
                {
                  
                  V_n = ( (nObservations - nMissingThisColumn) + 1./100. );

                  mCovariates = ( (nObservations - nMissingThisColumn) * covariatesOnlyMean(j) /* + ZERO/100.  */ ) / V_n;

                  aCovariates = 1.001 + 0.5 * (double)nObservations;
                  bCovariates = 100 + 0.5 * ( (nObservations - nMissingThisColumn - 1) * covariatesOnlyVar(j) +
                   ( ( (nObservations - nMissingThisColumn) * 1./100. ) / V_n ) * (covariatesOnlyMean(j)*covariatesOnlyMean(j)) );

                  vSquareCovariates = bCovariates * ( 1. + V_n ) / (aCovariates * V_n ) ;

                  tmpVec = Distributions::randT(nMissingThisColumn, 2. * aCovariates ) *
                      std::sqrt( vSquareCovariates ) + mCovariates; // % is element-wise multiplication
                  

                    if( varType( covariatesOnlyIdx(j) ) == 0 )
                    {
                        // variable is continuous, just impute tmpVec in
                        data.submat(missingIdxThisColumn,currentCol) = tmpVec;

                    }else if( varType( covariatesOnlyIdx(j) ) == 1 )
                    {
                        // variable is binary
                        for(unsigned int i=0; i<nMissingThisColumn; ++i)
                        {
                            if( std::abs( tmpVec(i) ) < std::abs( tmpVec(i)-1. ) )
                            {
                                data( missingIdxThisColumn(i), covariatesOnlyIdx(j) ) = 0;
                            }else{
                                data( missingIdxThisColumn(i), covariatesOnlyIdx(j) ) = 1;
                            }
                        }
                    }else if( varType( covariatesOnlyIdx(j) ) == 2 )
                    {
                        // variable ordinal, just impute a rounded tmpVec in
                        if( arma::any( arma::round(tmpVec) < arma::min(data.col( covariatesOnlyIdx(j) ))) )
                        {
                            tmpVec( arma::find( arma::round(tmpVec) < arma::min(data.col( covariatesOnlyIdx(j) )) ) ).fill( arma::min(data.col( covariatesOnlyIdx(j) )) );
                        }
                        if( arma::any( arma::round(tmpVec) > arma::max(data.col( covariatesOnlyIdx(j) )) ) )
                        {
                            tmpVec( arma::find( arma::round(tmpVec) > arma::max(data.col( covariatesOnlyIdx(j) )) ) ).fill( arma::max(data.col( covariatesOnlyIdx(j) )) );
                        }

                        // now impute
                        data.submat(missingIdxThisColumn, currentCol )  = arma::round(tmpVec);

                    }else{
                        // variable is of unknown type, assume continuous, just impute tmpVec in
                        data.submat(missingIdxThisColumn, currentCol )  = tmpVec;
        
                    }
                } // end this column
            } // end for each column



            // NOW ONTO THE Ys
            // they're all assumed continuous
            // We'd like to impute them from their respective regressions, so ...

            unsigned int nEquations = outcomesIdx.size();

            for( unsigned int k=0; k<nEquations; ++k)
            {
                // take the t/Normal distriution of the correspondent regression and input one by one all the outcomes.
                // They're assumed independent for now, so no big deal in doing them separately ...

                // see Sec 6.5 in bayesGauss.pdf and the likelihoodSUR function to get the parameters in C++
                // Sec 10 too look at their t-student parametrisation -- IMPORTANT IS if x~t_n(m,v^2) then (x-m)/v ~ t_n
                // so we can use the c++ sampler easily as in 
                // Distributions::randT(.,.) * v + m

                // m = arma::trans(X_new.cols(VS_IN)) * tilde_B
                // v^2 = b_r_n * (1. + X_new * W_n * X_new) / a_r_n
                // note that (from for ex http://blue.for.msu.edu/NEON/SC/slides/BayesianLinearRegression.pdf) we have that
                // b_r_n / a_r_n = s^2 = arma::as_scalar( arma::trans(data.col(outcomesIdx(l))) * data.col(outcomesIdx(l)) - ( tilde_B.t() * arma::inv_sympd(W_n) * tilde_B ) )
                // or something like that in the non-informative case

                // goryDetails.pdf give all the info needed

                // get the structure from the SEMEquation thinghy

                arma::vec mPredictive, vSquarePredictive, aPredictive, bPredictive;
                arma::uvec VS_IN, xVS_IN;
                arma::vec tilde_B; arma::mat W_n; double a_r_n,b_r_n;
                arma::mat X_new, XtX;

                arma::uvec nonMissingIdxThisColumn;

                for( unsigned int j=0, nOutcomes = outcomesIdx[k].n_elem; j<nOutcomes; ++j)
                {
                    currentCol = outcomesIdx[k](j);

                    // check howManyNaNInThisColumn
                    missingIdxThisColumn = missingDataIdxArray.submat( arma::find( missingDataIdxArray.col(1) == outcomesIdx[k](j) ) , arma::zeros<arma::uvec>(1) ); // arma::zeros<arma::uvec>(1) is basically one '0', but in uvec shape
                    nMissingThisColumn = missingIdxThisColumn.n_elem;

                    nonMissingIdxThisColumn = Utils::arma_setdiff_idx( arma::regspace<arma::uvec>(0,nObservations-1) , missingIdxThisColumn );

                    if( nMissingThisColumn > 0 )
                    {
                        
                        VS_IN = arma::join_vert( fixedPredictorsIdx[k] , 
                                                vsPredictorsIdx[k]( find( gamma_state[k].slice(0).col(j) != 0 ) ) );
        
                        xVS_IN = arma::join_vert( arma::regspace<arma::uvec>(0,fixedPredictorsIdx[k].n_elem-1) ,  // the fixed part
                                            fixedPredictorsIdx[k].n_elem + find( gamma_state[k].slice(0).col(j) != 0 ) );  // the VS part
                    
                        XtX = arma::trans( data.cols( VS_IN ) ) * data.cols( VS_IN );

                        W_n = arma::inv_sympd( XtX + arma::inv_sympd( W_0[k](xVS_IN,xVS_IN) ) );
                        tilde_B = W_n * ( arma::trans( data.submat(nonMissingIdxThisColumn,VS_IN) ) * data.submat(nonMissingIdxThisColumn, currentCol )  /* + W_0[k].i() * ZERO  */ );

                        a_r_n = a_r_0 + 0.5 * (double)( nObservations - nMissingThisColumn) ;
                        b_r_n = b_r_0 + 0.5 * arma::as_scalar( arma::trans(data.submat(nonMissingIdxThisColumn, currentCol )) * data.submat(nonMissingIdxThisColumn, currentCol ) - 
                                ( tilde_B.t() * arma::inv_sympd(W_n) * tilde_B ) );
                    
                        X_new = data.submat(missingIdxThisColumn,VS_IN);

                        mPredictive = X_new * tilde_B;

                        vSquarePredictive = arma::vec(nMissingThisColumn);
                        for( unsigned int i=0; i<nMissingThisColumn; ++i)
                        {
                            vSquarePredictive(i) = b_r_n * (1. + arma::as_scalar( X_new.row(i) * W_n * (X_new.row(i)).t() ) ) / a_r_n ;
                        }

                        tmpVec = Distributions::randT(nMissingThisColumn, 2. * a_r_n ) % arma::sqrt( vSquarePredictive ) + mPredictive; // % is element-wise multiplication
                        data.submat(missingIdxThisColumn, currentCol ) = tmpVec;

                        // std::cout << j <<"  -- m:" << mPredictive(0) << " ! " << vSquarePredictive(0) << std::flush << std::endl;
                        
                        
                    }
                }

            }

        }

    }
}