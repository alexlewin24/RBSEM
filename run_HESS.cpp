#include "run_HESS.h"

int run_HESS(std::string inFile, std::string outFilePath, unsigned int nIter, unsigned int nChains, long long unsigned int seed = 0, int method=1)
{

	omp_init_lock(&RNGlock);  // RNG lock for the parallel part

	// ############# Read the data
	unsigned int n,p;
	arma::uvec s,NAIdx;
	arma::ivec blockLabel,varType;
	arma::mat data;

	if( Utils::readDataSEM(inFile, data, blockLabel, varType, NAIdx,
		s, p, n) ){
		std::cout << "Reading successfull!" << std::endl;
	}else{
		std::cout << "OUCH! EXITING --- " << std::endl;
		return 1;
	}

	unsigned int nBlocks = s.n_elem;

	// Add to X the intercept
	arma::uword tmpUWord = arma::as_scalar(arma::find(blockLabel == 0,1,"first"));
	data.insert_cols( tmpUWord , arma::ones<arma::vec>(n) );
	blockLabel.insert_cols( tmpUWord, arma::zeros<arma::ivec>(1) ); // add its blockLabel

	// now find the indexes for each block in a more armadillo-interpretable way
	std::vector<arma::uvec> blockIdx(nBlocks+1);
	for( unsigned int k = 0; k<(nBlocks+1); ++k)
	{
		blockIdx[k] = arma::find(blockLabel == 0);
	}

	// XtX covariates only
	arma::mat XtX = data.cols(blockIdx[0]).t() * data.cols(blockIdx[0]); // this is needed for crossover for example
	arma::mat covariatesCorrelation = arma::inv( arma::diagmat( arma::sqrt(XtX.diag()) ) ) * XtX * arma::inv( arma::diagmat( arma::sqrt(XtX.diag()) ) );


	// ############# Init the RNG generator/engine
	std::random_device r;
	unsigned int nThreads = omp_get_max_threads();
	if(nChains < nThreads)
		nThreads = nChains;
	omp_set_num_threads(nThreads);

	rng.reserve(nThreads);  // reserve the correct space for the vector of rng engines
	std::seed_seq seedSeq;	// and declare the seedSequence
	std::vector<unsigned int> seedInit(8);
	long long int seed = r();

	// seed all the engines
	for(unsigned int i=0; i<nThreads; ++i)
	{
		// seedInit = {r(), r(), r(), r(), r(), r(), r(), r()};
		// seedSeq.generate(seedInit.begin(), seedInit.end()); // init with a sequence of 8 TRUE random values
		rng[i].seed(seed + i*(1000*(p*s*3+s*s)*nIter)); // 1000 rng per (p*s*3+s*s) variables each loop .. is this...ok? is random better?
		// rng[i].seed(r());
	}


	// ############################
	// HESS is the model where we have a (possibly) multivariate response with diagonal covariance matrix but we do VS separately for each outcome
	// Beta is a matrix of (1+p)xq coefficients, R is a diagonal qxq matrix (or a qx1 vector of sigma coeffs) and gamma is a pxq matrix
	// We sample using a MH step for gamma while Beta and R are integrated out
	// Then we repeat for each block in the SEM model
	// ############################

	// ### Set hyper-parameters of distributions involved

	// ## Set Prior Distributions  -- (and init values)

	// # R_k -- diagonal matrix with elements coming from IG(a_r_0,b_r_0)
	double a_r_0 = 0.1;
	double b_r_0 = 0.01;
	// these could be a vector for each k, but is it worth it? (TODO)


	// # Beta_k s --- (Alpha s, intercepts, are included here even though there's no variable selection on them)
	// for k>1 this includes the prior on the lambdas as well, since they have no different treatment! (TODO)
	// Beta comes from a Matrix Normal distribution with parameters MN(mean_beta_0,W_0,I_sk)   I_sk could be R_k, but is it worth it?

	double m_beta_0 = 0;  // this are common for all the coefficients
	double sigma2_beta_0 = 10;

	// arma::mat mean_beta_0(p+1, s); mean_beta_0.fill(m_beta_0); // this is assumed zero everywhere. STOP
	std::vector<arma::mat> W_0(nBlocks);
	unsigned int tmpSize = 0;
	for( unsigned int k=0; k<nBlocks; ++k)
	{
		tmpSize = p+1;
		if( k>0 )
			tmpSize += arma::sum( s(arma::span(0,k)) );

		W_0[k] = arma::eye( tmpSize, tmpSize ) * sigma2_beta_0;   // there's really no point in passing these matrices instead of the single coeff, but still...legacy code (TODO)
	}


	// # gamma  (p+1xs elements, but the first row doesn't move, always 1, cause are the intercepts)
	// this is taken into account further on in the algorithm, so no need to take into account now
	// REMEMBER TO CHAIN X AND Y_k's (k'<k) with X FIRST for each k so that intercept is always first
	// gamma_jk (j=1,..p  i.e. not 0 - k=0,s-1) comes from a Bernoulli distribution of parameters omega_j, who are all coming from a Beta(a_0,b_0)
	arma::vec a_0(p); a_0.fill( 5. ); 		// the average of a beta is E[p]=a/(a+b), this way E[p] <= 1.% FOR EVERY OUTCOME
	arma::vec b_0(p); b_0.fill( std::max( (double)p , 500.) - 5. );
	// again this could be different for every block, depending on the actual number of coeffs, but... (TODO)

	// ### Initialise and Start the chain

	// ## Defines proposal parameters and temporary variables for the MCMC
	arma::vec accCount = arma::zeros<arma::vec>(nBlocks); // one for each block
	arma::vec accCount_tmp = arma::zeros<arma::vec>(nChains-1); // this is instead just one for each extra chain

	// ## Initialise chain traces and states

	arma::mat omega_init;
	arma::umat gamma_init;

	std::vector<arma::mat> omega_curr(nBlocks);				//current state of the main chain
	std::vector<arma::umat> gamma_curr(nBlocks);
	arma::vec logLik_curr(nBlocks);
	arma::vec logPrior_curr(nBlocks);

	std::vector<arma::cube> omega_state;				// Current state of ALL the chains, we need them for the global moves
	std::vector<arma::ucube> gamma_state;
	std::vector<arma::vec> logPrior_state;
	std::vector<arma::vec> logLik_state;

	arma::uvec tmpPredictorIdx;

	for( unsigned int k=0; k<nBlocks; ++k)
	{

		tmpSize = W_0[k].n_cols -1 ;  // no intercept

		omega_init = arma::ones<arma::mat>(tmpSize,s(k)) / static_cast<double>(tmpSize);
		gamma_init = arma::zeros<arma::umat>(tmpSize,s(k));

		// TODO, these could be read from file, but how? for each block?	
		// if( omegaInitPath == "" )
		// if( gammaInitPath == "" )
		// 	omega_init.load(omegaInitPath,arma::raw_ascii);
		// 	for(unsigned int j=0; j<p; ++j)
		// 		for(unsigned int l=0; l<s(k); ++l)
		// 			gamma_init(j,l) = Distributions::randBernoulli(omega_init(j,l));
	

		omega_state[k] = arma::cube(tmpSize,s(k),nChains); 	
		gamma_state[k] = arma::ucube(tmpSize,s(k),nChains);	
		logPrior_state[k] = arma::vec(nChains);
		logLik_state[k] = arma::vec(nChains);

		logPrior_state[k](0) = Model::logSURPrior(omega_init, gamma_init, a_0, b_0);

		tmpPredictorIdx = blockIdx[0];
		for( unsigned int l=0; l<k; ++l)
			tmpPredictorIdx.insert_rows(tmpPredictorIdx.n_elem, blockIdx[l+1]);

		logLik_state[k](0) = Model::logSURLikelihood(data.cols(blockIdx[k+1]), data.cols(tmpPredictorIdx), gamma_init, a_r_0, b_r_0, W_0, 1.);


	}

	// HERE

	for(unsigned int m=0; m<nChains ; ++m)
	{
		omega_state.slice(m) = mcmc_omega.slice(0);
		gamma_state.slice(m) = mcmc_gamma.slice(0);
		logPrior_state(m) = mcmc_logPrior(0);
		logLik_state(m) = mcmc_logLik(0);
	}

	// std::cout << std::endl<< std::endl<< Model::logSURLikelihood(Y, X, gamma_init, a_r_0, b_r_0, W_0, 1.)<< std::endl;


	// # Define temperture ladder
	double maxTemperature = arma::min(a_0) - 2.; // due to the tempered gibbs moves
	double temperatureRatio = 2.; // std::max( 100., (double)n );

	arma::vec temperature(nChains);
	temperature(0) = 1.;

	for(unsigned int m=1; m<nChains; ++m)
		temperature(m) = std::min( maxTemperature, temperature(m-1)*temperatureRatio );

	for(unsigned int m=0; m<nChains ; ++m)
	{
		logLik_state(m) = mcmc_logLik(0)/temperature(m);
	}

	unsigned int nGlobalUpdates = floor( nChains/2 );

	unsigned int countGlobalUpdates = 0;    // count the global updates that happen on the first chain
	double accCountGlobalUpdates = 0.;		// count the accepted ones

	unsigned int globalType;

	// All-exchange operator hyper pars

	// Crossover operator hyper pars (see http://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n21.pdf
	double pXO_0 = 0.1, pXO_1 = 0.2 , pXO_2 = 0.2;
	double p11 = pXO_0*pXO_0 + (1.-pXO_0)*(1.-pXO_0) ,p12 = 2.*pXO_0*(1.-pXO_0) ,p21= pXO_1*(1.-pXO_2) + pXO_2*(1.-pXO_1) ,p22 = pXO_1*pXO_2 + (1.-pXO_1)*(1.-pXO_2);


	// NON-BANDIT related things
	unsigned int nUpdates = p/10; //arbitrary nunmber, should I use something different?

	// BANDIT ONLY SECTION
	arma::cube alpha_z;	arma::cube beta_z;	arma::cube zeta; std::vector<arma::vec> mismatch;
	std::vector<arma::vec> normalised_mismatch; std::vector<arma::vec> normalised_mismatch_backwards;

	if( method == 1)
	{
		nUpdates = 4; // for Bandit this must be SMALL (as it scales with nUpdates! and has a different meaning anyway)

		// Starting vaues for the Bandit tuning parameters
		// stating proposals are beta( 0.5 , 0.5 ) so that they're centered in 0.5 with spikes at the extremes
		// ALL OF THESE NEED TO HAVE A COPY FOR EACH CHAIN!!
		alpha_z = arma::cube(p,s,nChains); alpha_z.fill(0.5);
		beta_z = arma::cube (p,s,nChains); beta_z.fill(0.5);

		// these do not, as they will be overwritten anyway, but a copy helps in avoiding parallel access and/or OMP overhead in creating privates
		zeta = arma::cube(p,s,nChains);
		mismatch = std::vector<arma::vec>(nChains);
		normalised_mismatch = std::vector<arma::vec>(nChains);
		normalised_mismatch_backwards = std::vector<arma::vec>(nChains);
		for(unsigned int i = 0; i<nChains; ++i)
		{
			mismatch[i] = arma::vec(p*s);
			normalised_mismatch[i] = arma::vec(p*s);
			normalised_mismatch_backwards[i] = arma::vec(p*s);
		}
		// I still wonder why .slice() is an instance of arma::mat and return a reference to it
		// while .col() is a subview and has hald the method associated with a Col object ...
	}
	// END Bandit only section

	// ###########################################################
	// ###########################################################
	// ## Start the MCMC
	// ###########################################################
	// ###########################################################

	std::cout << "Starting "<< nChains <<" (parallel) chain(s) for " << nIter << " iterations:" << std::endl;

	for(unsigned int i=1; i < nIter ; ++i)
	{

		switch(method){

			case 0:
					#pragma omp parallel for num_threads(nThreads)
					for(unsigned int m=0; m<nChains ; ++m)
					{
							Model::MC3_SUR_MCMC_step(Y,X, omega_state.slice(m),gamma_state.slice(m),logPrior_state(m),logLik_state(m),
											a_r_0, b_r_0, W_0, a_0, b_0, accCount_tmp(m), nUpdates, temperature(m)); // in ESS accCount could be a vec, one for each chain
					}// end parallel updates
					break;

			case 1:
					#pragma omp parallel for num_threads(nThreads)
					for(unsigned int m=0; m<nChains ; ++m)
					{
							Model::bandit_SUR_MCMC_step(Y,X, omega_state.slice(m),gamma_state.slice(m),logPrior_state(m),logLik_state(m),
											a_r_0, b_r_0, W_0, a_0, b_0, accCount_tmp(m), nUpdates, temperature(m) ,
											zeta.slice(m), alpha_z.slice(m), beta_z.slice(m),
											mismatch[m], normalised_mismatch[m], normalised_mismatch_backwards[m]); // in ESS accCount could be a vec, one for each chain
					}// end parallel updates
					break;

					// there is no default since by default method = 2 and hence this last one is the default anyway
		}


		// UPDATE OUTPUT STATE
		mcmc_omega.slice(i) = omega_state.slice(0); mcmc_gamma.slice(i) = gamma_state.slice(0); 
		mcmc_logPrior(i) = logPrior_state(0); mcmc_logLik(i) = logLik_state(0);

		// ####################
		// ## Global moves
		if( nChains > 1 )
		{
			for(unsigned int k=0; k < nGlobalUpdates ; ++k)  // repeat global updates
			{

				// # Global move
				// Select the type of exchange/crossOver to apply

				globalType = Distributions::randIntUniform(0,6);   // (nChains-1)*(nChains-2)/2 is the number of possible chain combinations with nChains

				switch(globalType){

					case 0: break;

					// -- Exchange
					case 1: Model::exchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
							temperature, nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;

					case 2: Model::nearExchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
							temperature, nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;

					case 3: Model::allExchange_step(gamma_state, omega_state, logPrior_state, logLik_state,
							temperature, nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;

					// -- CrossOver
					case 4: Model::adapCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
								a_r_0, b_r_0, W_0, a_0, b_0, Y, X, temperature, 
								pXO_0, pXO_1, pXO_2, p11, p12, p21, p22,
								nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;

					case 5: Model::uniformCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
								a_r_0, b_r_0, W_0, a_0, b_0, Y, X, temperature,
								nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;

					case 6: Model::blockCrossOver_step(gamma_state, omega_state, logPrior_state, logLik_state,
								a_r_0, b_r_0, W_0, a_0, b_0, Y, X, covariatesCorrelation, temperature, 0.25,
								nChains, nGlobalUpdates, accCountGlobalUpdates, countGlobalUpdates);
							break;
				}


			} // end "K" Global Moves

			// Update the output vars
			mcmc_omega.slice(i) = omega_state.slice(0);
			mcmc_gamma.slice(i) = gamma_state.slice(0);
			mcmc_logPrior(i) = logPrior_state(0);
			mcmc_logLik(i) = logLik_state(0);

			// ## Update temperature ladder
			if ( i % 10 == 0 )
			{
				if( (accCountGlobalUpdates / (double)countGlobalUpdates) > 0.35 )
				{
					temperatureRatio = std::max( 2. , temperatureRatio-deltaTempRatio );
				}else{
					temperatureRatio += deltaTempRatio;
				}

				temperatureRatio = std::min( temperatureRatio , pow( maxTemperature, 1./( (double)nChains - 1.) ) );

				// std::cout << "---------------------------------- \n"<< temperature << '\n'<< std::flush;
				for(unsigned int m=1; m<nChains; ++m)
				{
					// untempered lik and prior
					logLik_state(m) = logLik_state(m)*temperature(m);
					logPrior_state(m) = logPrior_state(m)*temperature(m);

					// std::cout << '\n'<< temperature(m-1) << " " << temperature(m-1)*temperatureRatio  << '\n' << '\n'<< std::flush;
					temperature(m) = std::min( maxTemperature, temperature(m-1)*temperatureRatio );

					// re-tempered lik and prior
					logLik_state(m) = logLik_state(m)/temperature(m);
					logPrior_state(m) = logPrior_state(m)/temperature(m);

				}
			}


		} //end Global move's section

		// Print something on how the chain is going
		if( (i+1) % 100 == 0 )
		{
			// Update Acc Rate only for main chain
			accCount = accCount_tmp(0)/nUpdates;

			std::cout << " Running iteration " << i+1 << " ... loc.acc.rate ~ " << accCount/(double)i;
			if( nChains > 1 )
				std::cout << " global.acc.rate ~ " << accCountGlobalUpdates / (double)countGlobalUpdates;
			std::cout << /*"  ~~  " << temperature.t() << */ std::endl;
		}

	} // end MCMC


	std::cout << " MCMC ends. Final temperature ratio ~ " << temperatureRatio << "   --- Saving results and exiting" << std::endl;

	// ### Collect results and save them

	std::size_t slash = inFile.find("/");  // remove the path from inFile
	while( slash != std::string::npos )
	{
		inFile.erase(inFile.begin(),inFile.begin()+slash+1);
		slash = inFile.find("/");
		// std::cout << inFile << std::endl;
	}
	inFile.erase(inFile.end()-4,inFile.end());  // remomve the .txt from inFile
		// std::cout << inFile << std::endl;

	mcmc_omega.save(outFilePath+inFile+"_HESS_omega_out.txt",arma::raw_ascii);
	mcmc_gamma.save(outFilePath+inFile+"_HESS_gamma_out.txt",arma::raw_ascii);
	mcmc_logPrior += mcmc_logLik; mcmc_logPrior.save(outFilePath+inFile+"_HESS_logP_out.txt",arma::raw_ascii);


	// Exit
	return 0;
}
