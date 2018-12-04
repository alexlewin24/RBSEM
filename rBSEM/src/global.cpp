#include "global.h"

#ifdef _OPENMP
omp_lock_t RNGlock; // GLOBAL NAMESPACE CAUSE I NEED IT NO MATTER WHERE
#endif

std::vector<std::mt19937_64> rng;
