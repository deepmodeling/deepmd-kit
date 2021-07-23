#include "Gaussian.h"

void 
Gaussian::
set_seed (unsigned long s) 
{
  RandomGenerator_MT19937:: init_genrand (s);
}

void
Gaussian::
gen (double * vec, const int numb_gen)
{
  const double epsilon = std::numeric_limits<double>::min();
  const double two_pi = 2.0*M_PI;

  for (int ii = 0; ii < numb_gen; ++ii){
    double u0, u1;
    do {
      u0 = RandomGenerator_MT19937::genrand_real3();
      u1 = RandomGenerator_MT19937::genrand_real3();
    } while (u0 < epsilon);
    vec[ii] = sqrt(-2.0 * log(u0)) * cos(two_pi * u1);
  }
}


