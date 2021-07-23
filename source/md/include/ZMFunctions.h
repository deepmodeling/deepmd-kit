#ifndef __Functions_h_ZM_wanghan__
#define __Functions_h_ZM_wanghan__

#include <vector>
using namespace std;

namespace ZeroMultipole {
  double funcV (const double & alpha,
		const double & r);
  double funcD1V (const double & alpha,
		  const double & r);
  double funcD2V (const double & alpha,
		  const double & r);
  double funcD3V (const double & alpha,
		  const double & r);
  double funcD4V (const double & alpha,
		  const double & r);

  void calCoefficients (const int & ll,
			const double & alpha,
			const double & rc,
			vector<double > & coeff);

  class Potential
  {
    double		alpha, rc;
    int			ll;
    vector<double >	coeff;
public:
    Potential ();
    Potential (const int & ll,
	       const double & alpha,
	       const double & rc);
    void reinit (const int & ll,
		 const double & alpha,
		 const double & rc);
    double pot (const double & rr);
    double ulpot (const double & rr);
    double mpotp (const double & rr);
    double mulpotp (const double & rr);
public:
    double energyCorr (const vector<double > & charges) const;
  }
      ;
}



#endif

