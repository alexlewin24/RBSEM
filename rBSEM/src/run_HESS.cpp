#include "run_HESS.h"
// #include <boost/filesystem.hpp>  // can't use it as it doesn't work in R

int run_HESS(std::string inFile, std::string outFilePath, unsigned int nIter, unsigned int nChains, long long unsigned int seed, int method)
{

	omp_init_lock(&RNGlock);  // init RNG lock for the parallel part

	// ###########################################################
	// ###########################################################
	// ## Read Arguments and Data
	// ###########################################################
	// ###########################################################

	// ############# Read the data
	unsigned int n,p;
	arma::uvec s,NAIdx;
	arma::ivec blockLabel,varType;
	arma::mat data;   // here are ALL the data

	if( Utils::readDataSEM(inFile, data, blockLabel, varType, NAIdx,
		s, p, n) ){
		std::cout << "Reading successfull!" << std::endl;
	}else{
		std::cout << "OUCH! EXITING --- " << std::endl;
		return 1;
	}

	unsigned int nEquations = s.n_elem;
	unsigned int nBlocks = s.n_elem+1;

	// Add to X the intercept
	arma::uword tmpUWord = arma::as_scalar(arma::find(blockLabel == 0,1,"first"));
	data.insert_cols( tmpUWord , arma::ones<arma::vec>(n) );
	blockLabel.insert_rows( tmpUWord, arma::zeros<arma::ivec>(1) ); // add its blockLabel

	// now find the indexes for each block in a more armadillo-interpretable way
	std::vector<arma::uvec> blockIdx(nBlocks);
	for( unsigned int k = 0; k<(nBlocks); ++k)
	{
		blockIdx[k] = arma::find(blockLabel == k);
	}

	// XtX covariates only
	arma::mat XtX;
	std::vector<arma::mat> covariatesCorrelation(nEquations);
	arma::uvec predictorsIdx;
	
	for(unsigned int k=0; k<nEquations; ++k)
	{
	  predictorsIdx = blockIdx[0];
	  for(unsigned int j=1; j<=k; ++j)
	    predictorsIdx = arma::join_cols(predictorsIdx,blockIdx[j]); // is there a more efficient way to do it? TODO
	  
	  XtX = data.cols(predictorsIdx).t() * data.cols(predictorsIdx); // this is needed for crossover for example
	  covariatesCorrelation[k] = arma::inv( arma::diagmat( arma::sqrt(XtX.diag()) ) ) * XtX * arma::inv( arma::diagmat( arma::sqrt(XtX.diag()) ) );
	}

	// ############# Init the RNG generator/engine
	std::random_device r;
	unsigned int nThreads = omp_get_max_threads();
	if(nChains < nThreads)
		nThreads = nChains;
	omp_set_num_threads(nThreads);

	rng = std::vector<std::mt19937_64>(nThreads); //.reserve(nThreads);  // reserve the correct space for the vector of rng engines
	std::seed_seq seedSeq;	// and declare the seedSequence
	std::vector<unsigned int> seedInit(8);
	long long unsigned int seed_init; // = r();  // we could make it random

	// seed all the engines
	for(unsigned int i=0; i<nThreads; ++i)
	{
		// seedInit = {r(), r(), r(), r(), r(), r(), r(), r()};
		// seedSeq.generate(seedInit.begin(), seedInit.end()); // init with a sequence of 8 TRUE random values
		seed_init = seed; // + static_cast<long long unsigned int>(i*(1000*(p*arma::accu(s)*3+s*s)*nIter));
		rng[i] = std::mt19937_64( seed_init ); // 1000 rng per (p*s*3+s*s) variables each loop .. is this...ok? is random better?
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
	std::vector<arma::mat> W_0(nEquations);
	unsigned int tmpSize = 0;

	// # gamma  (p+1xs elements, but the first row doesn't move, always 1, cause are the intercepts)
	// this is taken into account further on in the algorithm, so no need to take into account now
	// REMEMBER TO CHAIN X AND Y_k's (k'<k) with X FIRST for each k so that intercept is always first
	// gamma_jk (j=1,..p  i.e. not 0 - k=0,s-1) comes from a Bernoulli distribution of parameters omega_j, who are all coming from a Beta(a_0,b_0)
	std::vector<arma::vec> a_0(nEquations);
	std::vector<arma::vec> b_0(nEquations);

	for( unsigned int k=0; k<nEquations; ++k)
	{
		tmpSize = p+1;
		if( k>0 )
			tmpSize += arma::sum( s(arma::span(0,k-1)) );

		W_0[k] = arma::eye( tmpSize, tmpSize ) * sigma2_beta_0;   // there's really no point in passing these matrices instead of the single coeff, but still...legacy code (TODO)
		// notice moreover that this assumes that each block is regressed on ALL the previous ones

		a_0[k] = arma::vec(tmpSize); a_0[k].fill( 5. ); 		// the average of a beta is E[p]=a/(a+b), this way E[p] <= 1.% FOR EVERY OUTCOME
		b_0[k] = arma::vec(tmpSize); b_0[k].fill( std::max( (double)p , 500.) - 5. );
		// again this could be different for every block, depending on the actual number of coeffs, but... (TODO)

	}

	
	// ### Initialise and Start the chain

	// ## Defines proposal parameters and temporary variables for the MCMC
	arma::vec accCount = arma::zeros<arma::vec>(nEquations); // one for each block
	arma::mat accCount_tmp = arma::zeros<arma::mat>(nEquations,nChains); // this is instead one for each block and each chain

	// ## Initialise chain traces and states

	std::vector<arma::mat> omega_init(nEquations);
	std::vector<arma::umat> gamma_init(nEquations);

	std::vector<arma::mat> omega_curr(nEquations);				//current state of the main chain
	std::vector<arma::umat> gamma_curr(nEquations);
	arma::vec logLik_curr(nEquations);
	arma::vec logPrior_curr(nEquations);

	std::vector<arma::cube> omega_state(nEquations);				// Current state of ALL the chains, we need them for the global moves
	std::vector<arma::ucube> gamma_state(nEquations);
	std::vector<arma::vec> logPrior_state(nEquations);
	std::vector<arma::vec> logLik_state(nEquations);

	arma::uvec tmpPredictorIdx;

	for( unsigned int k=0; k<nEquations; ++k)
	{

		tmpSize = W_0[k].n_cols -1 ;  // no intercept

		// different for each block?
		omega_init[k] = arma::ones<arma::mat>(tmpSize,s(k)) / static_cast<double>(tmpSize);
		gamma_init[k] = arma::zeros<arma::umat>(tmpSize,s(k));

		// TODO, these could be read from file, but how? for each block?	
		// if( omegaInitPath == "" )
		// if( gammaInitPath == "" )
		// 	omega_init.load(omegaInitPath,arma::raw_ascii);
		// 	for(unsigned int j=0; j<p; ++j)
		// 		for(unsigned int l=0; l<s(k); ++l)
		// 			gamma_init(j,l) = Distributions::randBernoulli(omega_init(j,l));
	
		// init
		omega_state[k] = arma::cube(tmpSize,s(k),nChains); 	
		gamma_state[k] = arma::ucube(tmpSize,s(k),nChains);	

		logPrior_state[k] = arma::vec(nChains);
		logLik_state[k] = arma::vec(nChains);

		// fill
		omega_state[k].slice(0) = omega_init[k];
		gamma_state[k].slice(0) = gamma_init[k];

		logPrior_state[k](0) = Model::logSURPrior(omega_init[k], gamma_init[k], a_0[k], b_0[k]);

		tmpPredictorIdx = blockIdx[0];
		for( unsigned int l=0; l<k; ++l)
			tmpPredictorIdx.insert_rows(tmpPredictorIdx.n_elem, blockIdx[l+1]);

		logLik_state[k](0) = Model::logSURLikelihood(data.cols(blockIdx[k+1]), data.cols(tmpPredictorIdx), gamma_init[k], a_r_0, b_r_0, W_0[k], 1.);

		for(unsigned int m=1; m<nChains ; ++m)
		{
			omega_state[k].slice(m) = omega_init[k];
			gamma_state[k].slice(m) = gamma_init[k];
			logPrior_state[k](m) = logPrior_state[k](0);
			logLik_state[k](m) = logLik_state[k](0);
		}
	}

	// # Define temperture ladder
	double maxTemperature = arma::min(a_0[0]) - 2.; // due to the tempered gibbs moves
	arma::vec temperatureRatio(nEquations); temperatureRatio.fill(2.); // std::max( 100., (double)n );
	double deltaTempRatio = 0.5;
	  
	std::vector<arma::vec> temperature(nEquations);
	for(unsigned int k=0; k<nEquations; ++k){
	  temperature[k] = arma::vec(nChains);
  	temperature[k](0) = 1.;
  
  	for(unsigned int m=1; m<nChains; ++m)
  		temperature[k](m) = std::min( maxTemperature, temperature[k](m-1)*temperatureRatio[k] );
	}

	for( unsigned int k=0; k<nEquations; ++k)
	{
		for(unsigned int m=1; m<nChains ; ++m)
		{
			logLik_state[k](m) = logLik_state[k](0)/temperature[k](m);
		}
	}
	
	unsigned int nGlobalUpdates = floor( nChains/2 );

	std::vector<unsigned int> countGlobalUpdates = arma::conv_to<std::vector<unsigned int>>::from(arma::zeros<arma::uvec>(nEquations));    // count the global updates that happen on the first chain
	std::vector<double> accCountGlobalUpdates = arma::conv_to<std::vector<double>>::from(arma::zeros<arma::vec>(nEquations));  // count the accepted ones

	// All-exchange operator hyper pars

	// Crossover operator hyper pars (see http://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n21.pdf
	double pXO_0 = 0.1, pXO_1 = 0.2 , pXO_2 = 0.2;
	double p11 = pXO_0*pXO_0 + (1.-pXO_0)*(1.-pXO_0) ,p12 = 2.*pXO_0*(1.-pXO_0) ,p21= pXO_1*(1.-pXO_2) + pXO_2*(1.-pXO_1) ,p22 = pXO_1*pXO_2 + (1.-pXO_1)*(1.-pXO_2);

	arma::vec parCrossOver(7);
	parCrossOver(0) = pXO_0;	parCrossOver(1) = pXO_1;	parCrossOver(2) = pXO_2;
	parCrossOver(3) = p11;	parCrossOver(4) = p12;	parCrossOver(5) = p21;	parCrossOver(6) = p22;

	// NON-BANDIT related things
	unsigned int nUpdates = p/10; //arbitrary nunmber, should I use something different?

	// BANDIT ONLY SECTION
	// arma::cube alpha_z;	arma::cube beta_z;	arma::cube zeta; std::vector<arma::vec> mismatch;
	// std::vector<arma::vec> normalised_mismatch; std::vector<arma::vec> normalised_mismatch_backwards;

	// if( method == 1)
	// {
	// 	nUpdates = 4; // for Bandit this must be SMALL (as it scales with nUpdates! and has a different meaning anyway)

	// 	// Starting vaues for the Bandit tuning parameters
	// 	// stating proposals are beta( 0.5 , 0.5 ) so that they're centered in 0.5 with spikes at the extremes
	// 	// ALL OF THESE NEED TO HAVE A COPY FOR EACH CHAIN AND BLOCK!! HUGE :/
	// 	alpha_z = arma::cube(p,s,nChains); alpha_z.fill(0.5);
	// 	beta_z = arma::cube (p,s,nChains); beta_z.fill(0.5);

	// 	// these do not, as they will be overwritten anyway, but a copy helps in avoiding parallel access and/or OMP overhead in creating privates
	// 	zeta = arma::cube(p,s,nChains);
	// 	mismatch = std::vector<arma::vec>(nChains);
	// 	normalised_mismatch = std::vector<arma::vec>(nChains);
	// 	normalised_mismatch_backwards = std::vector<arma::vec>(nChains);
	// 	for(unsigned int i = 0; i<nChains; ++i)
	// 	{
	// 		mismatch[i] = arma::vec(p*s);
	// 		normalised_mismatch[i] = arma::vec(p*s);
	// 		normalised_mismatch_backwards[i] = arma::vec(p*s);
	// 	}
	// 	// I still wonder why .slice() is an instance of arma::mat and return a reference to it
	// 	// while .col() is a subview and has hald the method associated with a Col object ...
	// }
	// END Bandit only section

	// ###########################################################
	// ###########################################################
	// ## Init Files
	// ###########################################################
	// ###########################################################

	// //check if provided out dir exists and create it otherwise
	// boost::filesystem::path path(outFilePath);
	// if ( !boost::filesystem::exists(path) || !boost::filesystem::is_directory(path) )
	// {
	// 	std::cout << path << " does not exist, creating...\n";
	// 	boost::filesystem::create_directory(path);
	// }

	// Open up out files

	// Re-define inFile so that I can use it in the output
	std::size_t slash = inFile.find("/");  // remove the path from inFile
	while( slash != std::string::npos )
	{
		inFile.erase(inFile.begin(),inFile.begin()+slash+1);
		slash = inFile.find("/");
		// std::cout << inFile << std::endl;
	}
	inFile.erase(inFile.end()-4,inFile.end());  // remomve the .txt from inFile


	// open new files in append mode
	// std::ofstream omegaOutFile; omegaOutFile.open( outFilePath+inFile+"_SSUR_omega_out.txt" , std::ios_base::trunc); omegaOutFile.close();
	std::ofstream gammaOutFile; 

	// zero out files
	for( unsigned int k=0; k<nEquations; ++k)
	{
		gammaOutFile.open( outFilePath+inFile+"_HESS_gamma_"+std::to_string(k)+"_out.txt" , std::ios_base::trunc); 
		gammaOutFile.close();
	}

	// Variable to hold the output to file ( current state )
	std::vector<arma::umat> gamma_out(nEquations);
	for( unsigned int k=0; k<nEquations; ++k)
		gamma_out[k] = gamma_state[k].slice(0); // out var for the gammas

	// could use save but I need to sum and then normalise so I'd need to store another matrix for each...

	// first output
	for( unsigned int k=0; k<nEquations; ++k)
	{
		gammaOutFile.open( outFilePath+inFile+"_HESS_gamma_"+std::to_string(k)+"_out.txt" , std::ios_base::trunc);
		gammaOutFile << (arma::conv_to<arma::mat>::from(gamma_out[k])) << std::flush;
		gammaOutFile.close();
	}


	// ###########################################################
	// ###########################################################
	// ## Start the MCMC
	// ###########################################################
	// ###########################################################


	std::cout << "Starting "<< nChains <<" (parallel) chain(s) for " << nIter << " iterations:" << std::endl;

	for(unsigned int iteration=1; iteration < nIter ; ++iteration)
	{

		Model::SEM_MCMC_step(data,blockIdx, 
					omega_state, gamma_state, logPrior_state, logLik_state,
					a_r_0, b_r_0, W_0, a_0, b_0, accCount_tmp, nUpdates, temperature, 0, // method forced to zero now, TODO
					parCrossOver, covariatesCorrelation, nGlobalUpdates, countGlobalUpdates, accCountGlobalUpdates,
					maxTemperature, temperatureRatio, deltaTempRatio); 

		// UPDATE OUTPUT STATE
		for( unsigned int k=0; k<nEquations; ++k)
			gamma_out[k] += gamma_state[k].slice(0); // the result of the whole procedure is now my new mcmc point, so add that up
		
		// Print something on how the chain is going
		if( (iteration+1) % 100 == 0 )
		{
			// Update Acc Rate only for each (block's) main chain
			for( unsigned int k=0; k<nEquations; ++k)
				accCount(k) = accCount_tmp(k,0)/nUpdates;

			std::cout << " Running iteration " << iteration+1 << " ... loc.acc.rate ~ " << accCount.t()/(double)iteration /*<< std::endl*/ ; 
			// no need for extra endl as accCount.t() finishes with '\n'

			// TODO FIGURE OUT HOW TO REPORT THESE
			// if( nChains > 1 )
			// 	std::cout << " global.acc.rate ~ " << accCountGlobalUpdates / (double)countGlobalUpdates;
			// std::cout << /*"  ~~  " << temperature.t() << */ std::endl;
		}

		// Output to files every now and then
		if( (iteration+1) % (1000) == 0 )
		{
			for( unsigned int k=0; k<nEquations; ++k)
			{
				gammaOutFile.open( outFilePath+inFile+"_HESS_gamma_"+std::to_string(k)+"_out.txt" , std::ios_base::trunc);
				gammaOutFile << (arma::conv_to<arma::mat>::from(gamma_out[k]))/((double)iteration+1.0) << std::flush;
				gammaOutFile.close();
			}
		}

	} // end MCMC


	std::cout << " MCMC ends. Final temperature ratio ~ " << temperatureRatio << "   --- Saving results and exiting" << std::endl;

	// Output to files one last time
	for( unsigned int k=0; k<nEquations; ++k)
	{
		gammaOutFile.open( outFilePath+inFile+"_HESS_gamma_"+std::to_string(k)+"_out.txt" , std::ios_base::trunc);
		gammaOutFile << (arma::conv_to<arma::mat>::from(gamma_out[k]))/((double)nIter+1.0) << std::flush;
		gammaOutFile.close();
	}


	// Exit
	return 0;
}