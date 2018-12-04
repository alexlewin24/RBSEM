#include "utils.h"

#include <iostream>
#include <string>
#include <armadillo>
#include <cmath>

#include <limits>

namespace Utils{


	bool readData(std::string fileName, unsigned int &nOutcomes, unsigned int &nPredictors, unsigned int &nObservations, arma::mat &Y, arma::mat& X)
	{

		bool status = X.load(fileName,arma::raw_ascii);

		if(!status){ 
			std::cout<< "Somethign went wrong while reading "<<fileName<<std::flush<<std::endl;
			return false;
		}else{

			if(X.n_cols < (nOutcomes+nPredictors) ){
				std::cout<< "Less columns than the sum of the specified outcomes and predictors. Specify the correct numbers. "<<std::flush<<std::endl;
				return false;
			}

			Y = X.cols(0,nOutcomes-1);
			X.shed_cols(0,nOutcomes-1);

			nObservations = Y.n_rows;

			if( nPredictors < X.n_cols){
				X.shed_cols(nPredictors,X.n_cols-1);
				std::cout<< "More predictors read then what specified in p=" << nPredictors << " -- I kept the firsts and discarded the others." << std::endl;
				nPredictors = X.n_cols;
			}

			if( nOutcomes < Y.n_cols){
				Y.shed_cols(nOutcomes,Y.n_cols-1);
				std::cout<< "More outcomes read then what specified in q=" << nOutcomes << " -- I kept the firsts and discarded the others." << std::endl;
				nOutcomes = Y.n_cols;
			}
		}

		return true;
	}

	bool readDataSEM(std::string fileName, arma::mat &data, arma::ivec &blockIndexes, arma::ivec &varType,
                  arma::umat SEMGraph, arma::uvec &missingDataIndexes, unsigned int &nObservations,
                  arma::uvec &nOutcomes, arma::uvec &nPredictors, std::vector<arma::uvec>& SEMEquations)
	{

		bool status = data.load(fileName,arma::raw_ascii);
		arma::uvec tmpUVec, predictorsIdx;
		arma::ivec uniqueBlockIndexes;
		
		if(!status){ 
			std::cout<< "Somethign went wrong while reading "<<fileName<<std::flush<<std::endl;
			return false;
		}else{

			// checks on the blockIndexes
			// index 0 stands for the Xs, predictors
			// index 1+ are the upper-level outcomes
			// so we always need at least some zeros and some ones
			uniqueBlockIndexes = arma::unique(blockIndexes);
			
			if( arma::max( blockIndexes ) < 1 || uniqueBlockIndexes.n_elem < 2 ) // more indepth check would be length of positive indexes..
			{
				std::cout<< "You need to define at least two blocks -- Xs and Ys"<<std::flush<<std::endl;
				return false;
			}

				// std::cout << "blkIdx: " << blockIndexes.t() << std::endl;
			// all the columns with blockIndex = -1 can be deleted
			arma::uword shedIdx;
			while( arma::any( arma::find(blockIndexes < 0)) )
			{
				shedIdx = arma::as_scalar(arma::find(blockIndexes < 0 , 1 , "first"));
				
				if(varType.n_elem == data.n_cols)
				    varType.shed_row( shedIdx ); //shed the varType as well!

				// then shed the rest				
				data.shed_col( shedIdx );
				blockIndexes.shed_row( shedIdx ); //shed the blockIdx as well!
			}

			// miscellanea variables
			arma::uvec outcomeIdx = arma::find( arma::sum(SEMGraph,0)!=0 );
			unsigned int nEquations = outcomeIdx.n_elem;
			
			SEMEquations = std::vector<arma::uvec>(nEquations);
			
			nOutcomes = arma::uvec(nEquations); // init this -- note this is nOutcomes FOR EACH Mv REGRESSION! (!= nEquations)
			nPredictors = arma::uvec(nEquations);
			
			for( unsigned int i=0; i<nEquations; ++i )
			{
			  
        // init this outcome
				tmpUVec = arma::find( blockIndexes == outcomeIdx(i) );   // groups in the graph are ordered by their position in the blockList
				nOutcomes(i) = tmpUVec.n_elem;
				
				//find its predictors
				predictorsIdx = arma::find( SEMGraph.col(outcomeIdx(i)) );
                                  
        // init the equation
        SEMEquations[i] = arma::uvec(1+predictorsIdx.n_elem);
        SEMEquations[i](0) = outcomeIdx(i);
        SEMEquations[i](arma::span(1,predictorsIdx.n_elem)) = predictorsIdx;
                                
        // init nPredictors
        nPredictors(i) = 0;
        for(unsigned int j=0; j<predictorsIdx.n_elem; ++j)
        {
          tmpUVec = arma::find( blockIndexes == predictorsIdx(j) );
          nPredictors(i) += tmpUVec.n_elem;  
        }
				
			}

			nObservations = data.n_rows;

			// Now deal with NANs
			if( data.has_nan() )
			{
				missingDataIndexes = arma::find_nonfinite(data);
				data(missingDataIndexes).fill( arma::datum::nan );  // This makes all the ind values into valid armadillo NANs (should be ok even without, but..)
			}
			else
			{
				missingDataIndexes.set_size(0);
			}
		}

		return true;
	}


	// sgn is defined in the header in order for it to be visible

	double logspace_add(const arma::vec& logv)
	{

		if( logv.is_empty() )
			return std::numeric_limits<double>::lowest();
		double m;
		if( logv.has_inf() || logv.has_nan() ) // || logv.is_empty()
		{
			return logspace_add(logv.elem(arma::find_finite(logv)));
		}else{ 
			m = arma::max(logv);
			return m + std::log( (double)arma::as_scalar( arma::sum( arma::exp( - (m - logv) ) ) ) );
		}
	}

	double logspace_add(double a, double b)
	{

		if(a <= std::numeric_limits<float>::lowest())
      		return b;
    	if(b <= std::numeric_limits<float>::lowest())
      		return a;
    	return std::max(a, b) + std::log( (double)(1. + std::exp( (double)-std::abs((double)(a - b)) )));
	}

	arma::uvec arma_setdiff_idx(const arma::uvec& x, const arma::uvec& y){

		arma::uvec ux = arma::unique(x);
		arma::uvec uy = arma::unique(y);

		for (size_t j = 0; j < uy.n_elem; j++) {
			arma::uvec q1 = arma::find(ux == uy[j]);
			if (!q1.empty()) {
				ux.shed_row(q1(0));
			}
		}

		return ux;
	}

}