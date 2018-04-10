#ifndef UTILS
#define UTILS

#include <iostream>
#include <string>
#include <armadillo>

namespace Utils{

	bool readData(std::string fileName, unsigned int &nOutcomes, unsigned int &nPredictors, unsigned int &nObservations, arma::mat &Y, arma::mat& X);

	bool readDataSEM(std::string fileName, arma::mat &data, arma::ivec &blockIndexes, arma::ivec &variableType,
			arma::uvec &missingDataIndexes, arma::uvec &nOutcomes, unsigned int &nPredictors, unsigned int &nObservations);

	template <typename T> int sgn(T val)
	{
		return (T(0) < val) - (val < T(0));
	}


	double logspace_add(const arma::vec& logv);
	double logspace_add(double a,double b);

	
}

#endif