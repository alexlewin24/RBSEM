#include "run_HESS.h"

int main(int argc, char *  argv[])
{

	unsigned int nIter = 10; // default number of iterations
	unsigned int burnin = 0;
	unsigned int nChains = 1;
	double deltaTempRatio = 1.;
	int seed = 0;

	std::string inFile = "data.txt";
	std::string outFilePath = "";
	// std::string omegaInitPath = "";
	// std::string gammaInitPath = "";
	std::string gammaInit = "S";

	unsigned int method = 0; // TODO Defaul should be our novel "bandit" method

	/*
	0: MC^3 -- BASE algorithm, simple randow walker with add-delete and swap move
	1: Bandit -- Novel method
	*/

    // ### Read and interpret command line (to put in a separate file / function?)
    int na = 1;
    
    while(na < argc)
    {
		if ( 0 == strcmp(argv[na],"--method") || 0 == strcmp(argv[na],"--algorithm") || 0 == strcmp(argv[na],"--algo")  )
		{
			method = std::stoi(argv[++na]);
			if(method > 1 || method < 0 )
			{
				std::cout << "Invalid method argument ("<<method<<"), see README.md\nDefaulting to bandit sampler\n"<<std::flush;
				method = 0;
			}
			if (na+1==argc) break;
			++na;
		}
		else if ( 0 == strcmp(argv[na],"--nIter") )
		{
			nIter = atoi(argv[++na]);
			if (na+1==argc) break;
			++na;
		}
		else if ( 0 == strcmp(argv[na],"--burnin") )
		{
			burnin = atoi(argv[++na]);
			if (na+1==argc) break;
			++na;
		}
		else if ( 0 == strcmp(argv[na],"--gammaInit") )
		{
			gammaInit = ""+std::string(argv[++na]); // use the next
			if (na+1==argc) break; // in case it's last, break
			++na;
		}
		else if ( 0 == strcmp(argv[na],"--seed") )
		{
			seed = atoi(argv[++na]); // use the next
			if (na+1==argc) break;  // in case it's last, break
			++na;
		}
		else if ( 0 == strcmp(argv[na],"--nChains") )
		{
			nChains = std::stoi(argv[++na]); // use the next
			if (na+1==argc) break; // in case it's last, break
			++na; // otherwise augment counter
		}
		else if ( 0 == strcmp(argv[na],"--inFile") )
		{
			inFile = ""+std::string(argv[++na]); // use the next
			if (na+1==argc) break; // in case it's last, break
			++na; // otherwise augment counter
		}
		else if ( 0 == strcmp(argv[na],"--outFilePath") )
		{
			outFilePath = std::string(argv[++na]); // use the next
			if (na+1==argc) break; // in case it's last, break
			++na; // otherwise augment counter
		}
		// else if ( 0 == strcmp(argv[na],"--omegaInitPath") )
		// {
		// 	omegaInitPath = ""+std::string(argv[++na]); // use the next
		// 	if (na+1==argc) break; // in case it's last, break
		// 	++na; // otherwise augment counter
		// }
		// else if ( 0 == strcmp(argv[na],"--gammaInitPath") )
		// {
		// 	gammaInitPath = ""+std::string(argv[++na]); // use the next
		// 	if (na+1==argc) break; // in case it's last, break
		// 	++na; // otherwise augment counter
		// }
		else if ( 0 == strcmp(argv[na],"--deltaTempRatio") )
		{
			deltaTempRatio = std::stof(argv[++na]); // use the next
			if (na+1==argc) break; // in case it's last, break
			++na; // otherwise augment counter
		}
		else
    {
	    std::cout << "Unknown option: " << argv[na] << std::endl;
	    return(1); //this is exit if I'm in a function elsewhere
    }
    }//end reading from command line


	int status= run_HESS(inFile, outFilePath, true, gammaInit, nIter, burnin, nChains, seed, method, 1);

	// Exit
	return status;
}
