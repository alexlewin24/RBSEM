#include "global.h"
#include "distr.h"

#ifndef HESS
#define HESS

namespace Model{

	double logSURPrior(const arma::mat &omega, const arma::umat &gamma, const arma::vec& a_0, const arma::vec& b_0);

	double logSURLikelihood(const arma::mat&Y, const arma::mat &X, const arma::umat &gamma, 
						 const double a_r_0, const double b_r_0, const arma::mat &W_0, double temp);

	void banditProposal
	(
	arma::umat &gamma, arma::mat &zeta, arma::umat &gamma_prop, arma::mat& alpha_z, arma::mat& beta_z,
	arma::vec& mismatch, arma::vec& normalised_mismatch, arma::vec& normalised_mismatch_backwards,
	const arma::mat&Y, const arma::mat &X, arma::mat& omega_curr, double& logPrior_curr, double &logLik_curr,
	const double a_r_0, const double b_r_0, const arma::mat& W_0, const arma::vec& a_0, const arma::vec& b_0,
	double& accCount, unsigned int nUpdates, double temp
	);


	void bandit_SUR_MCMC_step(const arma::mat&Y, const arma::mat &X, arma::mat& omega_curr, arma::umat& gamma_curr, double& logPrior_curr, double &logLik_curr, 
					const double a_r_0, const double b_r_0, const arma::mat& W_0, 
					const arma::vec& a_0, const arma::vec& b_0, double& accCount, unsigned int nUpdates, double temp,
					arma::mat& zeta, arma::mat& alpha_z, arma::mat& beta_z, arma::vec& mismatch,
					arma::vec& normalised_mismatch, arma::vec& normalised_mismatch_backwards);

	void MC3_SUR_MCMC_step(const arma::mat&Y, const arma::mat &X, arma::mat& omega_curr, arma::umat& gamma_curr, double& logPrior_curr, double &logLik_curr, 
					const double a_r_0, const double b_r_0, const arma::mat& W_0, 
					const arma::vec& a_0, const arma::vec& b_0, double& accCount, unsigned int nUpdates, double temp);

	void MCMC_Global_step(const arma::mat&Y, const arma::mat &X, unsigned int thisBlockIdx,
          arma::cube& omega_state, arma::ucube& gamma_state, 
          arma::vec& logPrior_state, arma::vec& logLik_state, 
          const double a_r_0, const double b_r_0, const arma::mat& W_0, 
          const arma::vec& a_0, const arma::vec& b_0, 
          const arma::vec& pCrossOver, const std::vector<arma::mat>& covariatesCorrelation, // tuning pars
          const unsigned int nChains, const unsigned int nGlobalUpdates,					// hyper tuning pars
          double& accCountGlobalUpdates, unsigned int& countGlobalUpdates,
          const arma::vec& temperature);

	void SEM_MCMC_step(const arma::mat& data, std::vector<arma::uvec> blockIdx,
          std::vector<arma::cube>& omega_state, std::vector<arma::ucube>& gamma_state, 
          std::vector<arma::vec>& logPrior_state, std::vector<arma::vec>& logLik_state,
          const double a_r_0, const double b_r_0, const std::vector<arma::mat>& W_0, 
          const std::vector<arma::vec>& a_0, const std::vector<arma::vec>& b_0,
          arma::mat& accCount, unsigned int nUpdates, std::vector<arma::vec>& temp, int method,
          const arma::vec& parCrossOver, const std::vector<arma::mat>& covariatesCorrelation,
          const unsigned int nGlobalUpdates, 
          std::vector<unsigned int>& countGlobalUpdates, std::vector<double>& accCountGlobalUpdates,
          const double maxTemperature, arma::vec& temperatureRatio, const double deltaTempRatio);
	

	void exchange_step(arma::ucube& gamma_state, arma::cube& omega_state, 
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

	void nearExchange_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

	void allExchange_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    				// states
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates,// tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

	void uniformCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
						const double a_r_0, const double b_r_0, const arma::mat& W_0,
						const arma::vec& a_0, const arma::vec& b_0, const arma::mat& Y,const arma::mat& X,	// Prior pars, data
						const arma::vec& temperature, const unsigned int nChains, const unsigned int nGlobalUpdates, // hyper tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

	void blockCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state,
						arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
						const double a_r_0, const double b_r_0, const arma::mat& W_0,
						const arma::vec& a_0, const arma::vec& b_0,const arma::mat& Y,const arma::mat& X,
						const arma::mat& covariatesCorrelation, const arma::vec& temperature,	// Prior pars, data
						const double threshold, const unsigned int nChains, const unsigned int nGlobalUpdates,		// hyper tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

	void adapCrossOver_step(arma::ucube& gamma_state, arma::cube& omega_state, 
						arma::vec& logPrior_state, arma::vec& logLik_state,    	// states
						const double a_r_0, const double b_r_0, const arma::mat& W_0,
						const arma::vec& a_0, const arma::vec& b_0,const arma::mat& Y,const arma::mat& X,	// Prior pars, data
						const arma::vec& temperature, double pXO_0, double pXO_1, double pXO_2,
						double p11, double p12, double p21, double p22,									// tuning pars
						const unsigned int nChains, const unsigned int nGlobalUpdates,					// hyper tuning pars
						double& accCountGlobalUpdates, unsigned int& countGlobalUpdates);

}

#endif