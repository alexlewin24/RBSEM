#include "global.h"
#include "utils.h"
#include "distr.h"
#include "imputation.h"
#include "HESS.h"
#include <armadillo>
#include <cmath>
#include <omp.h>

extern omp_lock_t RNGlock; //defined in global.h

namespace Model
{

	double logSURPrior(const arma::mat &omega, const arma::umat &gamma, const arma::vec& a_0, const arma::vec& b_0)
	{
		unsigned int p = gamma.n_rows;
		unsigned int s = gamma.n_cols;
		double logP = 0;

		for(unsigned int j=0; j<p; ++j)
		{
			for(unsigned int l=0; l<s; ++l)
			{
				logP += Distributions::logPDFBeta( omega(j,l) , a_0(j), b_0(j) );
				logP += Distributions::logPDFBernoulli( gamma(j,l), omega(j,l) );
			}	
		}

		return logP;
	}

	double logSURLikelihood(const arma::mat& data, const arma::uvec& outcomesIdx,
       const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
       const arma::umat &gamma, const double a_r_0, const double b_r_0, const arma::mat &W_0, double temp)
	{

		unsigned int n = data.n_rows;
		unsigned int s = outcomesIdx.n_elem;

		arma::uvec VS_IN;  // Variable Selection -- IN variables
		arma::uvec xVS_IN;  // Variable Selection -- IN variables AMONG THE Xs

		// define variables
		arma::mat W_n;
		arma::vec tilde_B;
		double a_r_n, b_r_n;
		double logP, sign, tmp; //sign is needed for the implementation, but we 'assume' that all the matrices are (semi-)positive-definite (-> det>=0)
		// tmp is a temporary container for the log determinants needed

		arma::mat XtX;

		logP = -log(M_PI)*((double)n*(double)s*0.5); // initialise with the normalising constant remaining from the likelhood

omp_set_lock(&RNGlock); // for some reason now I need to set a lock here or other threads will interfere.... 
// TODO, is it more beneficial to have parallel chains and sequential likelihood or viceversa?

		for(unsigned int l=0; l<s; ++l) // for each univaraite outcome ..
		{

			VS_IN = arma::join_vert( fixedPredictorsIdx , vsPredictorsIdx( find(gamma.col(l) != 0) ) );
	  
	        xVS_IN = arma::join_vert( arma::regspace<arma::uvec>(0,fixedPredictorsIdx.n_elem-1) ,  // the fixed part
                                fixedPredictorsIdx.n_elem + find(gamma.col(l) != 0) );  // the VS part
		  
			XtX = arma::trans( data.cols( VS_IN ) ) * data.cols( VS_IN );

			// hat_B = arma::inv_sympd(XtX) * arma::trans(X.cols(VS_IN)) * Y.col(l);
			W_n = arma::inv_sympd( XtX + arma::inv_sympd( W_0(xVS_IN,xVS_IN) ) );
			tilde_B = W_n * ( arma::trans( data.cols(VS_IN) ) * data.col(outcomesIdx(l))  /* + W_0.i() * ZERO  */ );

			a_r_n = a_r_0 + 0.5 * (double)n;
			b_r_n = b_r_0 + 0.5 * arma::as_scalar( arma::trans(data.col(outcomesIdx(l))) * data.col(outcomesIdx(l)) - 
			  ( tilde_B.t() * arma::inv_sympd(W_n) * tilde_B ) );

			arma::log_det( tmp, sign, W_n );
			logP += 0.5*tmp; 

			arma::log_det(tmp, sign, W_0(xVS_IN,xVS_IN) );
			logP -= 0.5*tmp; 

			logP += a_r_0*log(b_r_0) - a_r_n*log(b_r_n);

			logP += std::lgamma(a_r_n) - std::lgamma(a_r_0);
		}

omp_unset_lock(&RNGlock);

		return logP/temp;

	}



  void banditProposal(arma::umat &gamma, arma::mat &zeta, arma::umat &gamma_prop, arma::mat& alpha_z, arma::mat& beta_z,
      	arma::vec& mismatch, arma::vec& normalised_mismatch, arma::vec& normalised_mismatch_backwards,
      	const arma::mat& data, const arma::uvec& outcomesIdx,
      	const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
      	arma::mat& omega_curr, double& logPrior_curr, double &logLik_curr,
      	const double a_r_0, const double b_r_0, const arma::mat& W_0, const arma::vec& a_0, const arma::vec& b_0,
      	double& accCount, unsigned int nUpdates, double temp)
	{


		unsigned int p = gamma.n_rows;
		unsigned int s = gamma.n_cols;
		unsigned int n = data.n_rows;
		double log_proposal = 0.;

		double banditIncrement = 0.5;
		double banditLimit = n;
		bool finiteAdaptation = true;
		double ABRatio;

		// Sample Zs
		for(unsigned int i=0; i<p; ++i)
		{
			for(unsigned int j=0; j<s; ++j)
			{
				zeta(i,j) = Distributions::randBeta(alpha_z(i,j),beta_z(i,j));
			}
		}

		// Create mismatch
		for(unsigned int i=0; i<(p*s); ++i)
		{
			mismatch(i) = (gamma(i)==0)?(zeta(i)):(1.-zeta(i));   //mismatch
		}

		// Normalise
		// mismatch = arma::log(mismatch); //logscale
		// normalised_mismatch = mismatch - Utils::logspace_add(mismatch);


		normalised_mismatch = mismatch / arma::as_scalar(arma::sum(mismatch));

		arma::uvec update_idx;


		if( Distributions::randU01() < 0.5 )   // one deterministic update
		{

			// Decide which to update
			update_idx = arma::zeros<arma::uvec>(1);

			update_idx = Distributions::randWeightedIndexSampleWithoutReplacement(p*s,normalised_mismatch); // sample the one

			// Update
			gamma_prop(update_idx) = 1 - gamma(update_idx); // deterministic, just switch

			// Compute log_proposal probabilities
			normalised_mismatch_backwards = mismatch;
			normalised_mismatch_backwards(update_idx) = 1. - normalised_mismatch_backwards(update_idx) ;
			// normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
			normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

			log_proposal = arma::as_scalar( arma::log( normalised_mismatch_backwards(update_idx) ) ) -
						 arma::as_scalar( arma::log( normalised_mismatch(update_idx) ) );

		}else{

			
			// L(nUpdates) random (bern) updates
			// Note that we make use of column indexing here for armadillo matrices
			

			log_proposal = 0.;
			// Decide which to update
			update_idx = arma::zeros<arma::uvec>(nUpdates);
			update_idx = Distributions::randWeightedIndexSampleWithoutReplacement(p*s,normalised_mismatch,nUpdates); // sample nUpdates indexes

			normalised_mismatch_backwards = mismatch; // copy for backward proposal


			// Update
			for(unsigned int i=0; i<nUpdates; ++i)
			{
				gamma_prop(update_idx(i)) = Distributions::randBernoulli(zeta(update_idx(i))); // random update

				if(gamma_prop(update_idx(i)) != gamma(update_idx(i)))
				{
					normalised_mismatch_backwards(update_idx(i)) = 1.- normalised_mismatch_backwards(update_idx(i));
					log_proposal += Distributions::logPDFBernoulli(gamma(update_idx(i)),zeta(update_idx(i))) -
						Distributions::logPDFBernoulli(gamma_prop(update_idx(i)),zeta(update_idx(i)));
				}
			}

			// Compute log_proposal probabilities
			// normalised_mismatch_backwards = normalised_mismatch_backwards - Utils::logspace_add(normalised_mismatch_backwards);
			normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

			log_proposal += Distributions::logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards,update_idx) -
				Distributions::logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch,update_idx);
		}

		// A/R step

		double logLik_prop = logSURLikelihood(data,outcomesIdx,fixedPredictorsIdx,vsPredictorsIdx,
                                 gamma_prop, a_r_0, b_r_0, W_0, temp);
		
		double logPrior_prop = logSURPrior(omega_curr, gamma, a_0, b_0);

		double logAccProb = (logLik_prop - logLik_curr) + (logPrior_prop - logPrior_curr) + log_proposal;

		if( Distributions::randLogU01() < logAccProb )
		{
			accCount += update_idx.n_elem; // /(double)nUpdates;
			gamma = gamma_prop;
			logLik_curr = logLik_prop;
			logPrior_curr = logPrior_prop;

		}

		for(arma::uvec::iterator iter = update_idx.begin(); iter != update_idx.end(); ++iter)
		{
			if( finiteAdaptation )
			{
				if( alpha_z(*iter) + beta_z(*iter) < banditLimit )
				{
					alpha_z(*iter) += banditIncrement * gamma(*iter);
					beta_z(*iter) += banditIncrement * (1-gamma(*iter));
				}			
				// else we're done adapting	
			}else{
				// update them
				alpha_z(*iter) += banditIncrement * gamma(*iter);
				beta_z(*iter) += banditIncrement * (1-gamma(*iter));

				// then renormalise them
				if( alpha_z(*iter) + beta_z(*iter) > banditLimit )
				{
					ABRatio = alpha_z(*iter) / ( alpha_z(*iter) + beta_z(*iter) );
					alpha_z(*iter) = banditLimit * ABRatio;
					beta_z(*iter) = banditLimit * (1. - ABRatio);
					// the problem with this is that it keeps "resetting" ..
				}
			}

		}

	}


	void bandit_SUR_MCMC_step(const arma::mat& data, const arma::uvec& outcomesIdx,
           const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
           arma::mat& omega_curr, arma::umat& gamma_curr, 
           double& logPrior_curr, double &logLik_curr, 
					const double a_r_0, const double b_r_0, const arma::mat& W_0, 
					const arma::vec& a_0, const arma::vec& b_0, double& accCount, unsigned int nUpdates, double temp,
					arma::mat& zeta, arma::mat& alpha_z, arma::mat& beta_z, arma::vec& mismatch,
					arma::vec& normalised_mismatch, arma::vec& normalised_mismatch_backwards)
	{
		// Initialise proposal
		// arma::mat omega_prop;
		arma::umat gamma_prop = gamma_curr;

		// unsigned int n = Y.n_rows;
		unsigned int p = gamma_curr.n_rows;
		unsigned int s = gamma_curr.n_cols;
		arma::imat predUpdate;       // which predictor are we updating?
		arma::uword outUpdate; 		 // which outcome?
		
		double logPrior_prop, logLik_prop;
		double logAccProb;

		predUpdate = arma::join_rows( Distributions::randIntUniform(nUpdates, 0,p-1) , Distributions::randIntUniform(nUpdates,0,s-1) );    // note that I might be updating multiple times the same coeff
		
		double a,b;

		// ## Update Omega
		// FULL CONDITIONAL (no sense in making it parallel for now as the RNG is still sequential)
		// note that a_0 and b_0 are vectors of size p, meaning there's one prior for each beta, regardless of the associated Y
		for(unsigned int k=0; k<s; ++k)
		{
			for(unsigned int j=0; j<p; ++j)
			{
				a = a_0(j) + gamma_curr(j,k); b =  b_0(j) + 1. - gamma_curr(j,k);
				omega_curr(j,k) = Distributions::randBeta( (a+1.)/temp - 1. , b/temp );
			}
		}
	
		// Update the prior
		logPrior_curr = logSURPrior(omega_curr, gamma_curr, a_0, b_0); // update the prior probability

		// ## Update Gamma -- Local move
		banditProposal(gamma_curr, zeta, gamma_prop, alpha_z, beta_z,
				mismatch, normalised_mismatch, normalised_mismatch_backwards,
				data,outcomesIdx,fixedPredictorsIdx,vsPredictorsIdx,
				omega_curr, logPrior_curr, logLik_curr, a_r_0, b_r_0, W_0, a_0, b_0,
				accCount, nUpdates, temp);

	}

  //const arma::mat&Y, const arma::mat &X, 
	void MC3_SUR_MCMC_step(const arma::mat& data, const arma::uvec& outcomesIdx,
          const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
	        arma::mat& omega_curr, arma::umat& gamma_curr, 
	        double& logPrior_curr, double &logLik_curr,
					const double a_r_0, const double b_r_0, const arma::mat& W_0, 
					const arma::vec& a_0, const arma::vec& b_0, double& accCount, unsigned int nUpdates, double temp)
	{
		
		// Initialise proposal
		// arma::mat omega_prop;
		arma::umat gamma_prop(arma::size(gamma_curr));

		// unsigned int n = Y.n_rows;
		unsigned int p = gamma_curr.n_rows;   // note that p now is just the nmber of vsPredictors essentially, not of the whole set
		unsigned int s = outcomesIdx.n_elem;
		arma::imat predUpdate;       // which predictor are we updating?
		arma::uword outUpdate; 		 // which outcome?
		
		double logLik_prop;
		// double logPrior_prop;
		double logAccProb;

		// not needed, I update ALL the omegas via gibbs
		// predUpdate = arma::join_rows( Distributions::randIntUniform(nUpdates, 0,p-1) , Distributions::randIntUniform(nUpdates,0,s-1) );    // note that I might be updating multiple times the same coeff
		
		double a,b;

		// ## Update Omega
		// FULL CONDITIONAL (no sense in making it parallel for now as the RNG is still sequential)
		// note that a_0 and b_0 are vectors of size p, meaning there's one prior for each beta, regardless of the associated Y
		for(unsigned int k=0; k<s; ++k)
		{
			for(unsigned int j=0; j<p; ++j)
			{
				a = a_0(j) + gamma_curr(j,k); b =  b_0(j) + 1. - gamma_curr(j,k);
				omega_curr(j,k) = Distributions::randBeta( (a+1.)/temp - 1. , b/temp );
			}
		}
		
		// Update the prior
		logPrior_curr = logSURPrior(omega_curr, gamma_curr, a_0, b_0); // update the prior probability
    // likelihood unchanged
    

		// ## Update Gamma -- Local move
		predUpdate = Distributions::randIntUniform(nUpdates,0,p-1);    // note that I might be updating multiple times the same coeff

		for(unsigned int j=0; j<nUpdates ; ++j)
		{
			gamma_prop = gamma_curr;

			// Decide which l (l=0,...s-1) to sample
			outUpdate = Distributions::randIntUniform(0,s-1);    //maybe I need to update more often here that I have a matrix?

			/// Sample Gamma (see http://www.genetics.org/content/suppl/2011/09/16/genetics.111.131425.DC1/131425SI.pdf pag 5)
			gamma_prop(predUpdate(j),outUpdate) = Distributions::randBernoulli( pow(omega_curr(predUpdate(j),outUpdate),1./temp) / ( pow(omega_curr(predUpdate(j),outUpdate),1./temp) + pow( 1. - omega_curr(predUpdate(j),outUpdate) , 1./temp) ) ); 
			
			if( gamma_prop(predUpdate(j),outUpdate) != gamma_curr(predUpdate(j),outUpdate) )
			{
				// Compute log Probability for the current point
				logLik_prop = logSURLikelihood(data,outcomesIdx,fixedPredictorsIdx,vsPredictorsIdx,
                                   gamma_prop, a_r_0, b_r_0, W_0, temp);

				// A/R   
				logAccProb = (logLik_prop - logLik_curr);

				if( Distributions::randLogU01() < logAccProb )
				{
					accCount += 1./(double)nUpdates;
					gamma_curr(predUpdate(j),outUpdate) = gamma_prop(predUpdate(j),outUpdate);				
					logLik_curr = logLik_prop;
					logPrior_curr = logSURPrior(omega_curr, gamma_curr, a_0, b_0); // update the prior probability
				}
			}
		} // finish nUpdates on Gamma
	}


	void exchange_step(arma::ucube& gamma_state, arma::cube& omega_state, 
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{


		double pExchange;
		unsigned int chainIdx, firstChain, secondChain;

		arma::mat omegaSwap;
		arma::umat gammaSwap; // TODO, there's really no need to use a temp matrix when only a temp double is needed to swap in a loop...
		double logPriorSwap;
		double logLikSwap;


		// Select the chains to swap
		chainIdx = (nChains>2) ? Distributions::randIntUniform(1, (nChains)*(nChains-1)/2 ) : 1;   // (nChains-1)*(nChains-2)/2 is the number of possible chain combinations with nChains

		for(unsigned int c=1; c<nChains; ++c)
		{
			for(unsigned int r=0; r<c; ++r)
			{
				if( (--chainIdx) == 0 ){
					firstChain = r;
					secondChain = c;
					break;
				}
			}
		}

		countGlobalUpdates++;

		// Swap probability
		pExchange = logLik_state(secondChain) * ( temperature(secondChain)/temperature(firstChain) - 1.) +   // the plus is correct, don't doubt
					logLik_state(firstChain) * ( temperature(firstChain)/temperature(secondChain) - 1.) ;

		// The prior doesn't come into play as it's not tempered and hence there's no effect from it

		// DEBUG CHECK
		if(firstChain == 0 || secondChain == 0)
		{
			// std::cout << std::exp(pExchange) << std::endl <<std::flush;
			// std::cout << "Current -- " << logLik_state(firstChain)  << "  --  " << logLik_state(secondChain) << "  ===  " << logLik_state(firstChain) + logLik_state(secondChain)  << std::endl <<std::flush;
			// std::cout << "Proposed -- " << logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain)  << "  --  " << logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain)  <<
			//  "  ===  " << logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain) + logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain) << std::endl <<std::flush;
			// std::cout << pExchange - (
			// logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain) + 
			// logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain) - 
			// logLik_state(firstChain) - logLik_state(secondChain) ) << std::endl <<std::flush;
			// std::cout << temperature(firstChain)  << "  --  " << temperature(secondChain) << std::endl <<std::flush;
			// int jnk; std::cin >> jnk;
			// return ;
		}


		// A/R
		if( Distributions::randLogU01() < pExchange )
		{

			omegaSwap = omega_state.slice(secondChain);
			gammaSwap = gamma_state.slice(secondChain);
			logPriorSwap = logPrior_state(secondChain);
			logLikSwap = logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain);

			omega_state.slice(secondChain) = omega_state.slice(firstChain);
			gamma_state.slice(secondChain) = gamma_state.slice(firstChain);

			logPrior_state(secondChain) = logPrior_state(firstChain);
			logLik_state(secondChain) =  logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain);

			omega_state.slice(firstChain) = omegaSwap;
			gamma_state.slice(firstChain) = gammaSwap;
			logPrior_state(firstChain) = logPriorSwap;
			logLik_state(firstChain) = logLikSwap;

			accCountGlobalUpdates++;	

		}


	} //end generic Exchange step

	void nearExchange_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{

		double pExchange;
		unsigned int chainIdx, firstChain, secondChain;

		arma::mat omegaSwap;
		arma::umat gammaSwap;
		double logPriorSwap;
		double logLikSwap;

		// Select the chains to swap

		if( nChains>2 )
		{
			firstChain = Distributions::randIntUniform(1, nChains-2 );  // so not the first (0) or last (nChains-1) indexes
			secondChain = ( Distributions::randU01() < 0.5 ) ? firstChain-1 : firstChain+1 ; // then select a neighbour
		}else{
			// with just 2 chains
			firstChain = 0;
			secondChain = 1;
		}


		// if( firstChain == 0 || secondChain == 0 )   // if the swap involves the main chain
		countGlobalUpdates++;

		// Swap probability
		pExchange = 	logLik_state(secondChain) * ( temperature(secondChain)/temperature(firstChain) - 1.) +   // the plus is correct, don't doubt
						logLik_state(firstChain) * ( temperature(firstChain)/temperature(secondChain) - 1.) ;

		// A/R
		if( Distributions::randLogU01() < pExchange )
		{

			omegaSwap = omega_state.slice(secondChain);
			gammaSwap = gamma_state.slice(secondChain);
						logPriorSwap = logPrior_state(secondChain);
			logLikSwap = logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain);

			omega_state.slice(secondChain) = omega_state.slice(firstChain);
			gamma_state.slice(secondChain) = gamma_state.slice(firstChain);

			logPrior_state(secondChain) = logPrior_state(firstChain);
			logLik_state(secondChain) =  logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain);

			omega_state.slice(firstChain) = omegaSwap;
			gamma_state.slice(firstChain) = gammaSwap;
			logPrior_state(firstChain) = logPriorSwap;
			logLik_state(firstChain) = logLikSwap;

			// if (firstChain == 0 || secondChain == 0 )
			accCountGlobalUpdates++;

		}
	} //end Near Exchange

	void allExchange_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{

		arma::vec pExchange( (nChains)*(nChains-1)/2 +1 );
		unsigned int swapIdx, firstChain, secondChain;

		arma::umat indexTable( pExchange.n_elem, 2);
		unsigned int tabIndex = 0;
		indexTable(tabIndex,0) = 0; indexTable(tabIndex,1) = 0;
		tabIndex++;

		for(unsigned int c=1; c<nChains; ++c)
		{
			for(unsigned int r=0; r<c; ++r)
			{
				indexTable(tabIndex,0) = r; indexTable(tabIndex,1) = c;
				tabIndex++;
			}
		}

		arma::mat omegaSwap;
		arma::umat gammaSwap;
		double logPriorSwap;
		double logLikSwap;

		countGlobalUpdates++;

		// Select the chains to swap
		tabIndex = 0;
		pExchange(tabIndex) = 0.; // these are log probabilities, remember!
		tabIndex++;

		#pragma omp parallel for private(tabIndex, firstChain, secondChain)
		for(tabIndex = 1; tabIndex <= ((nChains)*(nChains-1)/2); ++tabIndex)
		{

			firstChain = indexTable(tabIndex,0);
			secondChain  = indexTable(tabIndex,1);

			// Swap probability
			// pExchange(tabIndex) = logLik_state(firstChain) * temperature(firstChain) / temperature(secondChain) +
			// 	logLik_state(secondChain) * temperature(secondChain) / temperature(firstChain) -
			// 	logLik_state(firstChain) - logLik_state(secondChain);

			pExchange(tabIndex) = 	logLik_state(secondChain) * ( temperature(secondChain)/temperature(firstChain) - 1.) +   // the plus is correct, don't doubt
				logLik_state(firstChain) * ( temperature(firstChain)/temperature(secondChain) - 1.) ;


		}

		// normalise and cumulate the weights
		double logSumWeights = Utils::logspace_add(pExchange); // normaliser
		arma::vec cumulPExchange = arma::cumsum( arma::exp( pExchange - logSumWeights ) ); // this should sum to one

		// Now select which swap happens
		double val = Distributions::randU01();

		swapIdx = 0;
		while( val > cumulPExchange(swapIdx) )
		{
			swapIdx++;
		}

		if( swapIdx != 0 )
		{

			firstChain = indexTable(swapIdx,0);
			secondChain  = indexTable(swapIdx,1);

			accCountGlobalUpdates++;

			omegaSwap = omega_state.slice(secondChain);
			gammaSwap = gamma_state.slice(secondChain);
			logPriorSwap = logPrior_state(secondChain);
			logLikSwap = logLik_state(secondChain)*temperature(secondChain)/temperature(firstChain);

			omega_state.slice(secondChain) = omega_state.slice(firstChain);
			gamma_state.slice(secondChain) = gamma_state.slice(firstChain);

			logPrior_state(secondChain) = logPrior_state(firstChain);
			logLik_state(secondChain) =  logLik_state(firstChain)*temperature(firstChain)/temperature(secondChain);

			omega_state.slice(firstChain) = omegaSwap;
			gamma_state.slice(firstChain) = gammaSwap;
			logPrior_state(firstChain) = logPriorSwap;
			logLik_state(firstChain) = logLikSwap;
		}
		//else swapIdx = 0 means no swap at all

	} //end ALL Exchange


void uniformCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state,
                           arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
                           const arma::mat& data, const arma::uvec& outcomesIdx,
                           const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
                           const double a_r_0, const double b_r_0, const arma::mat& W_0,
                           const arma::vec& a_0, const arma::vec& b_0,	// Prior pars
                           const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates, // hyper tuning pars
                           double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{


		unsigned int p = vsPredictorsIdx.n_elem; 
		unsigned int s = gamma_state.slice(0).n_cols;

		double pCrossOver;

		double logPriorFirst;
		double logPriorSecond;
		double logLikFirst;
		double logLikSecond;

		arma::ucube gammaXO(p,s,2);
		arma::cube omegaXO(p,s,2);
		unsigned int chainIdx, firstChain, secondChain;

		// Select the chains to XO 
		chainIdx = (nChains>2) ? Distributions::randIntUniform(1, (nChains)*(nChains-1)/2 ) : 1;   // (nChains-1)*(nChains-2)/2 is the number of possible chain combinations with nChains

		for(unsigned int c=1; c<nChains; ++c)
		{
			for(unsigned int r=0; r<c; ++r)
			{
				if( (--chainIdx) == 0 ){
					firstChain = r;
					secondChain = c;
					break;
				}
			}
		}

		countGlobalUpdates++; // a global update is happening

		// Propose Crossover
		for(unsigned int j=0; j<p; ++j)
		{
			for(unsigned int l=0; l<s; ++l)
			{
				if( Distributions::randU01() < 0.5 )
				{
					gammaXO(j,l,0) = gamma_state(j,l,firstChain);
					gammaXO(j,l,1) = gamma_state(j,l,secondChain); // 1-gammaXO(j,l,0); why did I have this? TODO

					omegaXO(j,l,0) = omega_state(j,l,firstChain);
					omegaXO(j,l,1) = omega_state(j,l,secondChain);

				}else{
					gammaXO(j,l,0) = gamma_state(j,l,secondChain);
					gammaXO(j,l,1) = gamma_state(j,l,firstChain); // 1-gammaXO(j,l,0); why did I have this? TODO

					omegaXO(j,l,0) = omega_state(j,l,secondChain);
					omegaXO(j,l,1) = omega_state(j,l,firstChain);
				}
			}
		}

		// Probability of acceptance

		logPriorFirst = logSURPrior(omega_state.slice(firstChain), gammaXO.slice(0), a_0, b_0);
		logPriorSecond = logSURPrior(omega_state.slice(secondChain), gammaXO.slice(1), a_0, b_0);
		
		logLikFirst = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx ,
                     gammaXO.slice(0), a_r_0, b_r_0, W_0, temperature(firstChain));
		
		logLikSecond = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx ,
		                  gammaXO.slice(1), a_r_0, b_r_0, W_0, temperature(secondChain));

		pCrossOver = (	logPriorFirst + logLikFirst - logPrior_state(firstChain) - logLik_state(firstChain) ) + 
						(	logPriorSecond + logLikSecond - logPrior_state(secondChain) - logLik_state(secondChain) );

		pCrossOver += 0.;  // XO prop probability simmetric now

		// A/R
		if( Distributions::randLogU01() < pCrossOver )
		{
			gamma_state.slice(firstChain) = gammaXO.slice(0);
			omega_state.slice(firstChain) = omegaXO.slice(0);
			logLik_state(firstChain) = logLikFirst;

			gamma_state.slice(secondChain) = gammaXO.slice(1);
			omega_state.slice(secondChain) = omegaXO.slice(1);
			logLik_state(secondChain) = logLikSecond;

			accCountGlobalUpdates++;

		} // end CrossOver
	}


void blockCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state,
           arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
           const arma::mat& data, const arma::uvec& outcomesIdx,
           const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
           const double a_r_0, const double b_r_0, const arma::mat& W_0,
           const arma::vec& a_0, const arma::vec& b_0,
           const arma::mat& covariatesCorrelation, const arma::vec& temperature,	// Prior pars, data
           const double threshold, const unsigned int nChains, const unsigned int nGlobalUpdates,		// hyper tuning pars
           double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{

		unsigned int p = gamma_state.n_rows; // there's the intercept
		unsigned int s = gamma_state.n_cols;

		double pCrossOver;

		double logPriorFirst;
		double logPriorSecond;
		double logLikFirst;
		double logLikSecond;

		arma::ucube gammaXO(p,s,2);
		arma::cube omegaXO(p,s,2);

		unsigned int chainIdx, firstChain, secondChain;

		// Select the chains to XO
		chainIdx = (nChains>2) ? Distributions::randIntUniform(1, (nChains)*(nChains-1)/2 ) : 1;   // (nChains-1)*(nChains-2)/2 is the number of possible chain combinations with nChains

		for(unsigned int c=1; c<nChains; ++c)
		{
			for(unsigned int r=0; r<c; ++r)
			{
				if( (--chainIdx) == 0 ){
					firstChain = r;
					secondChain = c;
					break;
				}
			}
		}

		countGlobalUpdates++; // a global update is happening

		// Propose Crossover

		// Select the ONE index to foor the block
		unsigned int predIdx = Distributions::randIntUniform(0, p-1 ); // pred
		unsigned int outcIdx = Distributions::randIntUniform(0, s-1 ); // outcome

		arma::uvec covIdx = arma::find( arma::abs( covariatesCorrelation.row(predIdx) ) > threshold );  // this will include the original predIdx
		
		gammaXO.slice(0) = gamma_state.slice(firstChain);
		gammaXO.slice(1) = gamma_state.slice(secondChain);

		omegaXO.slice(0) = omega_state.slice(firstChain);
		omegaXO.slice(1) = omega_state.slice(secondChain);

		for(unsigned int j=0; j<covIdx.n_elem; ++j)
		{
			gammaXO(covIdx(j),outcIdx,0) = gamma_state(covIdx(j),outcIdx,secondChain);
			gammaXO(covIdx(j),outcIdx,1) = gamma_state(covIdx(j),outcIdx,firstChain);

			omegaXO(covIdx(j),outcIdx,0) = omega_state(covIdx(j),outcIdx,secondChain);
			omegaXO(covIdx(j),outcIdx,1) = omega_state(covIdx(j),outcIdx,firstChain);

		}
		
		// Probability of acceptance
		logPriorFirst = logSURPrior(omega_state.slice(firstChain), gammaXO.slice(0), a_0, b_0);
		logPriorSecond = logSURPrior(omega_state.slice(secondChain), gammaXO.slice(1), a_0, b_0);
		
		logLikFirst = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx, 
                                 gammaXO.slice(0), a_r_0, b_r_0, W_0, temperature(firstChain));
		
		logLikSecond = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx,
                                 gammaXO.slice(1), a_r_0, b_r_0, W_0, temperature(secondChain));

		pCrossOver = (	logPriorFirst + logLikFirst - logPrior_state(firstChain) - logLik_state(firstChain) ) + 
						(	logPriorSecond + logLikSecond - logPrior_state(secondChain) - logLik_state(secondChain) );

		pCrossOver += 0.;  // XO prop probability is weird, how do I compute it? Let's say is symmetric as is determnistic and both comes from the same covariatesCorrelation

		// A/R
		if( Distributions::randLogU01() < pCrossOver )
		{
			gamma_state.slice(firstChain) = gammaXO.slice(0);
			omega_state.slice(firstChain) = omegaXO.slice(0);
			logLik_state(firstChain) = logLikFirst;

			gamma_state.slice(secondChain) = gammaXO.slice(1);
			omega_state.slice(secondChain) = omegaXO.slice(1);
			logLik_state(secondChain) = logLikSecond;

			accCountGlobalUpdates++;

		} // end CrossOver
	}

 
void adapCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state,
                        arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
                        const arma::mat& data, const arma::uvec& outcomesIdx,
                        const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
                        const double a_r_0, const double b_r_0, const arma::mat& W_0,
                        const arma::vec& a_0, const arma::vec& b_0,	// Prior pars
                        const arma::vec& temperature, double pXO_0, double pXO_1, double pXO_2,
                        double p11, double p12, double p21, double p22,									// tuning pars
                        const unsigned int nChains, const unsigned int nGlobalUpdates,					// hyper tuning pars
                        double& accCountGlobalUpdates, unsigned int& countGlobalUpdates)
	{

		unsigned int p = gamma_state.n_rows;
		unsigned int s = gamma_state.n_cols;

		double pCrossOver;
		unsigned int n11,n12,n21,n22;

		double logPriorFirst;
		double logPriorSecond;
		double logLikFirst;
		double logLikSecond;
		
		arma::ucube gammaXO(p,s,2);
		unsigned int chainIdx, firstChain, secondChain;


		// Select the chains to XO
		chainIdx = (nChains>2) ? Distributions::randIntUniform(1, (nChains)*(nChains-1)/2 ) : 1;   // (nChains-1)*(nChains-2)/2 is the number of possible chain combinations with nChains

		for(unsigned int c=1; c<nChains; ++c)
		{
			for(unsigned int r=0; r<c; ++r)
			{
				if( (--chainIdx) == 0 ){
					firstChain = r;
					secondChain = c;
					break;
				}
			}
		}
		// if( firstChain == 0 || secondChain == 0 )
		countGlobalUpdates++;

		// Propose Crossover
		n11=0;n12=0;n21=0;n22=0;

		for(unsigned int j=0; j<p; ++j)
		{ 
			for(unsigned int l=0; l<s; ++l)
			{
				if ( gamma_state(j,l,firstChain) == gamma_state(j,l,secondChain) )
				{
					gammaXO(j,l,0) = gamma_state(j,l,firstChain);
					gammaXO(j,l,1) = gamma_state(j,l,secondChain);

					gammaXO(j,l,0) = ( Distributions::randU01() < pXO_0 )? 1-gammaXO(j,l,0) : gammaXO(j,l,0);
					gammaXO(j,l,1) = ( Distributions::randU01() < pXO_0 )? 1-gammaXO(j,l,1) : gammaXO(j,l,1);
					
					if( gammaXO(j,l,0) == gammaXO(j,l,1) )
						++n11;
					else
						++n12;
				}
				else
				{
					gammaXO(j,l,0) = gamma_state(j,l,firstChain); // check correctness TODO
					gammaXO(j,l,1) = gamma_state(j,l,secondChain);

					gammaXO(j,l,0) = ( Distributions::randU01() < pXO_1 )? 1-gammaXO(j,l,0) : gammaXO(j,l,0);
					gammaXO(j,l,1) = ( Distributions::randU01() < pXO_2 )? 1-gammaXO(j,l,1) : gammaXO(j,l,1);

					if( gammaXO(j,l,0) == gammaXO(j,l,1) )
						++n21;
					else
						++n22;
				}
			}
		}
		// Probability of acceptance

		logPriorFirst = logSURPrior(omega_state.slice(firstChain), gammaXO.slice(0), a_0, b_0);
		logPriorSecond = logSURPrior(omega_state.slice(secondChain), gammaXO.slice(1), a_0, b_0);
		
		logLikFirst = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx,
                                 gammaXO.slice(0), a_r_0, b_r_0, W_0, temperature(firstChain));
		logLikSecond = logSURLikelihood(data, outcomesIdx , fixedPredictorsIdx, vsPredictorsIdx,
                                  gammaXO.slice(1), a_r_0, b_r_0, W_0, temperature(secondChain));

		pCrossOver = (	logPriorFirst + logLikFirst - logPrior_state(firstChain) - logLik_state(firstChain) ) + 
						(	logPriorSecond + logLikSecond - logPrior_state(secondChain) - logLik_state(secondChain) );

		pCrossOver += (n11 * log( p11 ) + n12 * log( p12 ) + n21 * log( p21 ) + n22 * log( p22 ) )-  // CrossOver proposal probability FORWARD
						(n11 * log( p11 ) + n12 * log( p21 ) + n21 * log( p12 ) + n22 * log( p22 ) );  // XO prop probability backward (note that ns stays the same but changes associated prob)

		// A/R
		if( Distributions::randLogU01() < pCrossOver )
		{
			gamma_state.slice(firstChain) = gammaXO.slice(0);
			logPrior_state(firstChain) = logPriorFirst;
			logLik_state(firstChain) = logLikFirst;

			gamma_state.slice(secondChain) = gammaXO.slice(1);
			logPrior_state(secondChain) = logPriorSecond;
			logLik_state(secondChain) = logLikSecond;

			// If I'm crossOver'ing on the first chain update also the mcmc_* variables
			// if(firstChain == 0)
			// {
			accCountGlobalUpdates++;
			// }
		}

	} // end CrossOver


	void MCMC_Global_step(const arma::mat& data, const arma::uvec& outcomesIdx,
						const arma::uvec& fixedPredictorsIdx, const arma::uvec& vsPredictorsIdx,
						unsigned int thisBlockIdx,
						arma::cube& omega_state, arma::ucube& gamma_state, 
						arma::vec& logPrior_state, arma::vec& logLik_state,
						const double a_r_0, const double b_r_0, const arma::mat& W_0, 
						const arma::vec& a_0, const arma::vec& b_0, 
						const arma::vec& pCrossOver, const arma::mat& covariatesCorrelation, // tuning pars
						const unsigned int nChains, const unsigned int nGlobalUpdates,					          // hyper tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates,
						const arma::vec& temperature)
	{
	
	double pXO_0 = pCrossOver(0);
	double pXO_1 = pCrossOver(1);
	double pXO_2 = pCrossOver(2);
	double p11 = pCrossOver(3);
	double p12 = pCrossOver(4);
	double p21 = pCrossOver(5);
	double p22 = pCrossOver(6);
	
	unsigned int globalType;
	
	for(unsigned int k=0; k < nGlobalUpdates ; ++k)  // repeat global updates
	{
		// # Global move
		// Select the type of exchange/crossOver to apply
		
		globalType = Distributions::randIntUniform(0,6);
		
		switch(globalType){
		
		case 0: break;
		
		// -- Exchange
		case 1: Model::exchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
									temperature, nChains, nGlobalUpdates,
									accCountGlobalUpdates, countGlobalUpdates);
		break;
		
		case 2: Model::nearExchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
										temperature, nChains, nGlobalUpdates,
										accCountGlobalUpdates, countGlobalUpdates);
		break;
		
		case 3: Model::allExchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
										temperature, nChains, nGlobalUpdates,
										accCountGlobalUpdates, countGlobalUpdates);
		break;
		
		// -- CrossOver
		case 4: Model::uniformCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
									data, outcomesIdx, fixedPredictorsIdx, vsPredictorsIdx,
									a_r_0, b_r_0, W_0, a_0, b_0, temperature,
									nChains, nGlobalUpdates,
									accCountGlobalUpdates, countGlobalUpdates);
		break;
		
	case 5: Model::adapCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
									data, outcomesIdx, fixedPredictorsIdx, vsPredictorsIdx,
									a_r_0, b_r_0, W_0, a_0, b_0, temperature, 
									pXO_0, pXO_1, pXO_2, p11, p12, p21, p22,
									nChains, nGlobalUpdates,
									accCountGlobalUpdates, countGlobalUpdates);
		break;
		
		
		case 6: Model::blockCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
								data, outcomesIdx, fixedPredictorsIdx, vsPredictorsIdx,
								a_r_0, b_r_0, W_0, a_0, b_0, covariatesCorrelation, temperature, 0.25,
								nChains, nGlobalUpdates,
								accCountGlobalUpdates, countGlobalUpdates);
		break;
		}
		
	} // end "K" Global Moves
	
	}


	void SEM_MCMC_step(const arma::mat& data, const std::vector<arma::uvec>& outcomesIdx,
					const std::vector<arma::uvec>& fixedPredictorsIdx, const std::vector<arma::uvec>& vsPredictorsIdx,
					std::vector<arma::cube>& omega_state, std::vector<arma::ucube>& gamma_state, 
					std::vector<arma::vec>& logPrior_state, std::vector<arma::vec>& logLik_state,
					const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0, 
					const std::vector<arma::vec>& a_0, const std::vector<arma::vec>& b_0,
					arma::mat& accCount, unsigned int nUpdates, std::vector<arma::vec>& temp, 
					std::vector<arma::cube>& zeta, std::vector<arma::cube>& alpha_z, 
					std::vector<arma::cube>& beta_z, std::vector<arma::field<arma::vec>>& mismatch,
					std::vector<arma::field<arma::vec>>& normalised_mismatch, 
					std::vector<arma::field<arma::vec>>& normalised_mismatch_backwards,
					int method,
					const arma::vec& parCrossOver, const std::vector<arma::mat>& covariatesCorrelation,
					const unsigned int nGlobalUpdates,
					std::vector<unsigned int>& countGlobalUpdates, std::vector<double>& accCountGlobalUpdates,
					const double maxTemperature, arma::vec& temperatureRatio, const double deltaTempRatio)
	{
		// This function will take care of all the local AND global moves for all the chains

	  unsigned int nThreads = omp_get_max_threads();
	  unsigned int nChains = omega_state[0].n_slices;
	  unsigned int nEquations = outcomesIdx.size();
	  
	  for(unsigned int k=0; k < nEquations ; ++k){
	    
	      switch(method){
	      
	      case 0:
          	#pragma omp parallel for num_threads(nThreads)
			for(unsigned int m=0; m<nChains ; ++m)
			{
				Model::MC3_SUR_MCMC_step(data, outcomesIdx[k], fixedPredictorsIdx[k], vsPredictorsIdx[k],
								omega_state[k].slice(m),gamma_state[k].slice(m),
								logPrior_state[k](m),logLik_state[k](m),
								a_r_0, b_r_0, W_0[k], a_0[k], b_0[k], accCount(k,m), 
								nUpdates, temp[k](m));
			} // end parallel updates
          break; // end case 0 
	        
	      case 1:
          	#pragma omp parallel for num_threads(nThreads)
	        for(unsigned int m=0; m<nChains ; ++m)
	        {
            	Model::bandit_SUR_MCMC_step(data, outcomesIdx[k], fixedPredictorsIdx[k], vsPredictorsIdx[k],
                                  omega_state[k].slice(m),gamma_state[k].slice(m),
                                  logPrior_state[k](m),logLik_state[k](m),
                                  a_r_0, b_r_0, W_0[k], a_0[k], b_0[k], 
                                  accCount(k,m), nUpdates, temp[k](m),
                                  zeta[k].slice(m), alpha_z[k].slice(m), beta_z[k].slice(m), 
                                  mismatch[k](m), normalised_mismatch[k](m),
                                  normalised_mismatch_backwards[k](m));
          	} // end parallel updates
	        break; // end case 1
					
	      }
	      
	      // ####################
        // Global moves
        
        if(nChains>1)
        {
          Model::MCMC_Global_step(data, outcomesIdx[k], fixedPredictorsIdx[k], vsPredictorsIdx[k], k,
                         omega_state[k], gamma_state[k],logPrior_state[k], logLik_state[k],
                         a_r_0, b_r_0, W_0[k], a_0[k], b_0[k],
                         parCrossOver, covariatesCorrelation[k],
                         nChains, nGlobalUpdates,accCountGlobalUpdates[k], countGlobalUpdates[k],
                         temp[k]);
          
          
          // ## Update temperature ladder
          if ( Distributions::randIntUniform(0,9) == 0 ) //once every ten times on average
          {
            if( (accCountGlobalUpdates[k] / (double)countGlobalUpdates[k]) > 0.35 )
            {
              temperatureRatio(k) += deltaTempRatio;
            }else{
              temperatureRatio(k) = std::max( 1. , temperatureRatio(k)-deltaTempRatio );
            }
            
            temperatureRatio(k) = std::min( temperatureRatio(k) , pow( maxTemperature, 1./( (double)nChains - 1.) ) );
            
            for(unsigned int m=1; m<nChains; ++m)
            {
              // untempered lik and prior
              logLik_state[k](m) = logLik_state[k](m)*temp[k](m);
              logPrior_state[k](m) = logPrior_state[k](m)*temp[k](m);
              
              temp[k](m) = std::min( maxTemperature, temp[k](m-1)*temperatureRatio(k) );
              
              // re-tempered lik and prior
              logLik_state[k](m) = logLik_state[k](m)/temp[k](m);
              logPrior_state[k](m) = logPrior_state[k](m)/temp[k](m);
              
            }
          }
          
          
        } //end Global move's section
        
       } // end block update

	}


	std::vector<arma::mat> sampleBeta( arma::mat& data,
                    const std::vector<arma::uvec>& outcomesIdx, const std::vector<arma::uvec>& fixedPredictorsIdx, 
                    const std::vector<arma::uvec>& vsPredictorsIdx,
                    const std::vector<arma::ucube>& gamma_state, const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0 )
	{

		unsigned int nEquations = outcomesIdx.size();

		arma::vec mPredictive, vSquarePredictive, aPredictive, bPredictive;
		arma::uvec VS_IN, xVS_IN;
		arma::vec tilde_B; 
		arma::mat W_n; 
		double a_r_n,b_r_n;
		arma::mat XtX;
		arma::uvec currentCol(1);
		arma::uvec singleIdx_j(1);

		arma::uvec nonMissingIdxThisColumn;
			

		std::vector<arma::mat> beta(nEquations);

		for( unsigned int k=0; k<nEquations; ++k)
		{
			beta[k] = arma::zeros<arma::mat>(fixedPredictorsIdx[k].n_elem+vsPredictorsIdx[k].n_elem,outcomesIdx[k].n_elem);

			for( unsigned int j=0, nOutcomes = outcomesIdx[k].n_elem; j<nOutcomes; ++j)
			{
				currentCol(0) = outcomesIdx[k](j);
				singleIdx_j(0) = j;

				VS_IN = arma::join_vert( fixedPredictorsIdx[k] , 
						vsPredictorsIdx[k]( find( gamma_state[k].slice(0).col(j) != 0 ) ) );
				
				xVS_IN = arma::join_vert( arma::regspace<arma::uvec>(0,fixedPredictorsIdx[k].n_elem-1) ,  // the fixed part
						fixedPredictorsIdx[k].n_elem + find( gamma_state[k].slice(0).col(j) != 0 ) );  // the VS part

				XtX = arma::trans( data.cols( VS_IN ) ) * data.cols( VS_IN );

				W_n = arma::inv_sympd( XtX + arma::inv_sympd( W_0[k](xVS_IN,xVS_IN) ) );
				tilde_B = W_n * ( arma::trans( data.cols(VS_IN) ) * 
					data.cols( currentCol )  /* + W_0[k].i() * ZERO  */ );

				beta[k].submat(xVS_IN,singleIdx_j) = Distributions::randMvNormal( tilde_B , W_n );
			}
		}

		return beta;

	}


} // end namespace