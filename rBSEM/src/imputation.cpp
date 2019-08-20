#include "imputation.h"

Imputation::Imputation(arma::mat &data, const arma::uvec &completeCases,
                       const arma::umat &SEMGraph,
                       const std::vector<arma::uvec> &blockIdx,
                       const arma::ivec &varType) {
  unsigned int nBlocks = SEMGraph.n_cols;
  unsigned int nObservations = data.n_rows;
  // block labels of variables that appear only on the rhs
  arma::uvec blockCovariatesOnlyIdx(nBlocks);
  // indexes of variables that appear only on the rhs
  covariatesOnlyIdx = arma::uvec(data.n_cols);

  unsigned int blockCovLeft = 0;
  unsigned int covLeft = 0;

  arma::uvec tmpToAdd;

  for (unsigned int k = 0; k < nBlocks; ++k) {
    if (arma::all(SEMGraph.col(k) == 0)) {
      // this block is made of rhs-only variables
      blockCovariatesOnlyIdx(blockCovLeft++) = k;
      // add the variables indexes
      tmpToAdd = blockIdx[k];
      covariatesOnlyIdx.subvec(covLeft, covLeft + tmpToAdd.n_elem - 1) =
          tmpToAdd;
      covLeft += tmpToAdd.n_elem;

    } else {
      // this block is made of lhs variables (for some equations)
      // so no need to do anything, the model structure will tell us how to init
      // afterward..
    }
  }

  blockCovariatesOnlyIdx.resize(
      blockCovLeft);                  // resize it to correct dimension
  covariatesOnlyIdx.resize(covLeft);  // resize it to correct dimension

  covariatesOnlyIdx =
      arma::sort(covariatesOnlyIdx);  // sort it -- this way all indexes respect
                                      // the data matrix ordering

  // now on principle we'd like to impute this covariatesOnly via a (latent)
  // MvNormal of sort.
  covariatesOnlyMean = arma::vec(covariatesOnlyIdx.n_elem);

  arma::mat latentDataMatrix(data.cols(covariatesOnlyIdx));

  // compute the mean of the imputation distribution and transform binary values
  // in latent normal ones
  // for each element in covariatesOnlyIdx
  for (unsigned int j = 0; j < covariatesOnlyIdx.n_elem; ++j) {
    double mean = 0.0, delta = 0.0;
    unsigned int count = 0;

    for (unsigned int i = 0; i < nObservations; ++i) {
      if ((latentDataMatrix(i, j) ==
           latentDataMatrix(i, j)))  // (not) definition of a non-NaN
      {
        if (varType(covariatesOnlyIdx(j)) !=
            0)  // so not a continuous var -- > i.e. a binary one!
        {
          // Transform the elements from 0/1 to -1.96/1.96 (inverseCDF-like)
          if (latentDataMatrix(i, j) == 0) {
            latentDataMatrix(i, j) = -1.96;
          } else {
            latentDataMatrix(i, j) = 1.96;
          }
        }

        count++;
        delta = latentDataMatrix(i, j) - mean;
        mean = mean + delta / (double)count;
      }
    }
    covariatesOnlyMean(j) = mean;
  }

  // for the covariance use only complete cases instead
  covariatesOnlyVar = arma::mat(covariatesOnlyIdx.n_elem,
                                covariatesOnlyIdx.n_elem, arma::fill::zeros);
  for (auto const &i : completeCases)
    covariatesOnlyVar += (latentDataMatrix.row(i).t() - covariatesOnlyMean) *
                         (latentDataMatrix.row(i) - covariatesOnlyMean.t());
  covariatesOnlyVar /= completeCases.n_elem;

  // check diagonal elements, if anything is zero (can happen for binary
  // variables especially) just give it a very small diagonal element
  for (size_t j = 0; j < covariatesOnlyVar.n_cols; ++j) {
    if (covariatesOnlyVar(j, j) <= 0) covariatesOnlyVar(j, j) = 1e-4;
  }

  nCompleteCases = completeCases.n_elem;
}

void Imputation::imputeAll(arma::mat &data,
                           const arma::uvec &missingDataIndexes,
                           const arma::umat &missingDataIdxArray,
                           arma::ivec &varType,
                           const std::vector<arma::uvec> &outcomesIdx,
                           const std::vector<arma::uvec> &fixedPredictorsIdx,
                           const std::vector<arma::uvec> &vsPredictorsIdx,
                           const std::vector<arma::ucube> &gamma_state,
                           const double a_r_0, const double b_r_0,
                           const std::vector<arma::mat> &W_0) {
  unsigned int nObservations = data.n_rows;

  if (missingDataIdxArray.n_elem > 0) {
    // actually impute the values

    arma::uvec rowsWithMissingData = arma::unique(missingDataIdxArray.col(0));

    // start with the Covariates Only
    // for each row with missing values
    for (auto const &i : rowsWithMissingData) {
      arma::uvec currentRow(1);
      currentRow(0) = i;

      // check howManyNaNInThisRow -- but only in the covariates-only columns!
      arma::uvec missingIdxThisRow = arma::intersect(
          covariatesOnlyIdx, missingDataIdxArray.submat(
                                 arma::find(missingDataIdxArray.col(0) == i),
                                 arma::ones<arma::uvec>(1)));

      if (missingIdxThisRow.n_elem > 0) {
        arma::uvec nonMissingIdxThisRow =
            Utils::arma_setdiff_idx(covariatesOnlyIdx, missingIdxThisRow);

        // Get the relative positions of (non)missingIdxThisRow wrt
        // covariatesOnlyIdx
        arma::uvec focusLatentCols =
            Utils::arma_get_vec_idx(missingIdxThisRow, covariatesOnlyIdx);
        arma::uvec remainingLatentCols = Utils::arma_setdiff_idx(
            arma::regspace<arma::uvec>(0, covariatesOnlyIdx.n_elem - 1),
            focusLatentCols);

        // Assume we're using the conditional distributions as likelihoods
        // (which is not correct, but as conditionals of t-distr are not t...)
        // (vec(0),2,2,diag(100.)) in this notation
        // https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions

        constexpr double mu_0 = 0.;
        constexpr double k_0 = 2.;
        constexpr double nu_0 = 2.;
        constexpr double psi_0 = 100.;

        // Variance and mean of the conditional distribution
        arma::mat cachedMatrixOp =
            covariatesOnlyVar(focusLatentCols, remainingLatentCols) *
            arma::inv_sympd(
                covariatesOnlyVar(remainingLatentCols, remainingLatentCols));

        arma::mat cov_n =
            covariatesOnlyVar(focusLatentCols, focusLatentCols) -
            cachedMatrixOp *
                covariatesOnlyVar(remainingLatentCols, focusLatentCols);
        arma::vec mean_n =
            covariatesOnlyMean(focusLatentCols) +
            cachedMatrixOp * (data(currentRow, nonMissingIdxThisRow).t() -
                              covariatesOnlyMean(remainingLatentCols));

        // updated parameter of the sampling distribution
        double k_n = (nCompleteCases + k_0);
        arma::vec mu_n = (nCompleteCases * mean_n + mu_0 * k_0) / k_n;
        double nu_n = (nCompleteCases + nu_0);
        arma::mat psi_n = psi_0 + nCompleteCases * cov_n +
                          (k_0 * nCompleteCases) / (k_n) *
                              ((mean_n - mu_0) * (mean_n - mu_0).t());

        double df = nu_n - focusLatentCols.n_elem + .1;

        arma::vec newSample =
            Distributions::randMvT(df, mu_n, (k_n + 1.) / (k_n * df) * psi_n);
        /*
        {
          std::cout << "Imputation happening with values: (" << k_n << " "
                    << nu_n << " ** " << mu_n.t()
                    << std::endl
                    // << cov_n << std::endl
                    // << covariatesOnlyVar(focusLatentCols, focusLatentCols)
                    // << std::endl
                    // << cachedMatrixOp *
                    // covariatesOnlyVar(remainingLatentCols, focusLatentCols)
                    // << std::endl
                    << psi_n << std::endl
                    << ") --> " << newSample.t() << std::endl
                    << std::endl;
          char c;
          std::cin >> c;
        }
        */

        for (size_t j = 0; j < missingIdxThisRow.n_elem; ++j) {
          if (varType(missingIdxThisRow(j)) == 0) {
            // variable is continuous, just impute newSample in
            data(i, missingIdxThisRow(j)) = newSample(j);

          } else if (varType(missingIdxThisRow(j)) == 1) {
            // variable is binary
            if (newSample(j) < 0.) {
              data(i, missingIdxThisRow(j)) = 0;
            } else {
              data(i, missingIdxThisRow(j)) = 1;
            }
          } else {
            // variable is of unknown type, assume continuous, just impute
            // newSample in
            data(i, missingIdxThisRow(j)) = newSample(j);
          }
        }

      }  // end if we have covariatesOnly missing in this row
    }    // end this row

    // **** end imputation of covariatesOnly

    // NOW ONTO THE Ys
    // they're all assumed continuous
    // We'd like to impute them from their respective regressions, so ...

    unsigned int nEquations = outcomesIdx.size();

    for (unsigned int k = 0; k < nEquations; ++k) {
      // take the t/Normal distriution of the correspondent regression and input
      // one by one all the outcomes. They're assumed independent for now, so no
      // big deal in doing them separately ...

      // see Sec 6.5 in bayesGauss.pdf and the likelihoodSUR function to get the
      // parameters in C++ Sec 10 too look at their t-student parametrisation --
      // IMPORTANT IS if x~t_n(m,v^2) then (x-m)/v ~ t_n so we can use the c++
      // sampler easily as in Distributions::randT(.,.) * v + m

      // m = arma::trans(X_new.cols(VS_IN)) * tilde_B
      // v^2 = b_r_n * (1. + X_new * W_n * X_new) / a_r_n
      // note that (from for ex
      // http://blue.for.msu.edu/NEON/SC/slides/BayesianLinearRegression.pdf) we
      // have that b_r_n / a_r_n = s^2 = arma::as_scalar(
      // arma::trans(data.col(outcomesIdx(l))) * data.col(outcomesIdx(l)) - (
      // tilde_B.t() * arma::inv_sympd(W_n) * tilde_B ) ) or something like that
      // in the non-informative case

      // goryDetails.pdf give all the info needed

      // get the structure from the SEMEquation thinghy

      arma::vec mPredictive, vSquarePredictive, aPredictive, bPredictive;
      arma::uvec VS_IN, xVS_IN;
      arma::vec tilde_B;
      arma::mat W_n;
      double a_r_n, b_r_n;
      arma::mat X_new, XtX;

      arma::uvec nonMissingIdxThisColumn;

      for (unsigned int j = 0, nOutcomes = outcomesIdx[k].n_elem; j < nOutcomes;
           ++j) {
        arma::uvec currentCol(1);
        currentCol = outcomesIdx[k](j);

        // check howManyNaNInThisColumn
        arma::uvec missingIdxThisColumn = missingDataIdxArray.submat(
            arma::find(missingDataIdxArray.col(1) == outcomesIdx[k](j)),
            arma::zeros<arma::uvec>(
                1));  // arma::zeros<arma::uvec>(1) is basically one '0',
                      // but in uvec shape
        size_t nMissingThisColumn = missingIdxThisColumn.n_elem;

        nonMissingIdxThisColumn = Utils::arma_setdiff_idx(
            arma::regspace<arma::uvec>(0, nObservations - 1),
            missingIdxThisColumn);

        if (nMissingThisColumn > 0) {
          VS_IN = arma::join_vert(
              fixedPredictorsIdx[k],
              vsPredictorsIdx[k](find(gamma_state[k].slice(0).col(j) != 0)));

          xVS_IN = arma::join_vert(
              arma::regspace<arma::uvec>(
                  0, fixedPredictorsIdx[k].n_elem - 1),  // the fixed part
              fixedPredictorsIdx[k].n_elem +
                  find(gamma_state[k].slice(0).col(j) != 0));  // the VS part

          XtX = arma::trans(data.cols(VS_IN)) * data.cols(VS_IN);

          W_n = arma::inv_sympd(XtX + arma::inv_sympd(W_0[k](xVS_IN, xVS_IN)));
          tilde_B =
              W_n * (arma::trans(data.submat(nonMissingIdxThisColumn, VS_IN)) *
                     data.submat(nonMissingIdxThisColumn,
                                 currentCol) /* + W_0[k].i() * ZERO  */);

          a_r_n = a_r_0 + 0.5 * (double)(nObservations - nMissingThisColumn);
          b_r_n =
              b_r_0 +
              0.5 * arma::as_scalar(
                        arma::trans(
                            data.submat(nonMissingIdxThisColumn, currentCol)) *
                            data.submat(nonMissingIdxThisColumn, currentCol) -
                        (tilde_B.t() * arma::inv_sympd(W_n) * tilde_B));

          X_new = data.submat(missingIdxThisColumn, VS_IN);

          mPredictive = X_new * tilde_B;

          vSquarePredictive = arma::vec(nMissingThisColumn);
          for (unsigned int i = 0; i < nMissingThisColumn; ++i) {
            vSquarePredictive(i) = b_r_n *
                                   (1. + arma::as_scalar(X_new.row(i) * W_n *
                                                         (X_new.row(i)).t())) /
                                   a_r_n;
          }

          arma::vec newSamples =
              Distributions::randT(nMissingThisColumn, 2. * a_r_n) %
                  arma::sqrt(vSquarePredictive) +
              mPredictive;  // % is element-wise multiplication
          data.submat(missingIdxThisColumn, currentCol) = newSamples;

          // std::cout << j <<"  -- m:" << mPredictive(0) << " ! " <<
          // vSquarePredictive(0) << std::flush << std::endl;
        }
      }
    }
  }
}