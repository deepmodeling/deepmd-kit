#include "Interpolation.h"
#include <iterator>


void Interpolation::piece6OrderInterpol (const double & a,   const double & b,
					 const double & va,  const double & vb,
					 const double & da,  const double & db,
					 const double & dda, const double & ddb,
					 Poly & p)
{
  std::vector<Poly > standardPolys(6);
  for (unsigned i = 0; i < 6; ++i){
    standardPolys[i].getOrder() = 5;
    standardPolys[i].getCoeffs().resize(6);
  }
  standardPolys[0].getCoeffs()[0] = 1;
  standardPolys[0].getCoeffs()[1] = 0;
  standardPolys[0].getCoeffs()[2] = 0;
  standardPolys[0].getCoeffs()[3] = -10;
  standardPolys[0].getCoeffs()[4] = 15;
  standardPolys[0].getCoeffs()[5] = -6;

  standardPolys[1].getCoeffs()[0] = 0;
  standardPolys[1].getCoeffs()[1] = 0;
  standardPolys[1].getCoeffs()[2] = 0;
  standardPolys[1].getCoeffs()[3] = 10;
  standardPolys[1].getCoeffs()[4] = -15;
  standardPolys[1].getCoeffs()[5] = 6;
  
  standardPolys[2].getCoeffs()[0] = 0;
  standardPolys[2].getCoeffs()[1] = 1;
  standardPolys[2].getCoeffs()[2] = 0;
  standardPolys[2].getCoeffs()[3] = -6;
  standardPolys[2].getCoeffs()[4] = 8;
  standardPolys[2].getCoeffs()[5] = -3;

  standardPolys[3].getCoeffs()[0] = 0;
  standardPolys[3].getCoeffs()[1] = 0;
  standardPolys[3].getCoeffs()[2] = 0;
  standardPolys[3].getCoeffs()[3] = -4;
  standardPolys[3].getCoeffs()[4] = 7;
  standardPolys[3].getCoeffs()[5] = -3;
  
  standardPolys[4].getCoeffs()[0] = 0;
  standardPolys[4].getCoeffs()[1] = 0;
  standardPolys[4].getCoeffs()[2] = 0.5;
  standardPolys[4].getCoeffs()[3] = -1.5;
  standardPolys[4].getCoeffs()[4] = 1.5;
  standardPolys[4].getCoeffs()[5] = -0.5;

  standardPolys[5].getCoeffs()[0] = 0;
  standardPolys[5].getCoeffs()[1] = 0;
  standardPolys[5].getCoeffs()[2] = 0;
  standardPolys[5].getCoeffs()[3] = 0.5;
  standardPolys[5].getCoeffs()[4] = -1;
  standardPolys[5].getCoeffs()[5] = 0.5;

  std::vector<Poly > scaledPolys (6);
  double tmpa (1./(b - a));
  double tmpb (-a / (b - a));
  for (unsigned i = 0; i < 6; ++i){
    standardPolys[i].valueLinearPoly (tmpa, tmpb, scaledPolys[i]);
  }
  scaledPolys[2] *= 1./tmpa;
  scaledPolys[3] *= 1./tmpa;
  scaledPolys[4] *= 1./tmpa/tmpa;
  scaledPolys[5] *= 1./tmpa/tmpa;

  p.zero();
  p += (scaledPolys[0] *= va);
  p += (scaledPolys[1] *= vb);
  p += (scaledPolys[2] *= da);
  p += (scaledPolys[3] *= db);
  p += (scaledPolys[4] *= dda);
  p += (scaledPolys[5] *= ddb);

  return ;
}



void Interpolation::pieceLinearInterpol (const double & a,  const double & b, 
					 const double & va, const double & vb,
					 Poly & p)
{
  double k = (vb-va) / (b-a);
  std::vector<double > tmp (2, 0);
  tmp[0] += va;
  tmp[0] += k * (-a);
  tmp[1] = k;
  p.reinit (tmp);
}

void Interpolation::piecewiseLinear (const std::vector<double > & x,
				     const std::vector<double > & y,
				     PiecewisePoly & ps)
{
  std::vector<double >::const_iterator pxp1 = x.begin();
  std::vector<double >::const_iterator px = (pxp1 ++);
  std::vector<double >::const_iterator pyp1 = y.begin();
  std::vector<double >::const_iterator py = (pyp1 ++);
  ps.clear();
  Poly tmpp;
  for (; pxp1 != x.end(); ++ pxp1, ++pyp1, ++px, ++py){
    pieceLinearInterpol (*px, *pxp1, *py, *pyp1, tmpp);
    ps.get_x().push_back (*px);
    ps.get_p().push_back (tmpp);
  }
  ps.get_x().push_back (*px);
}  

void Interpolation::pieceSecondDerivativeInterpol (
    const double & a,  const double & b,
    const double & va, const double & vb,
    const double & dda,const double & ddb,
    Poly & p)
{
  std::vector<double > tmp (2, 0);
  double k = (vb-va) / (b-a);
  tmp[0] += va;
  tmp[0] += k * (-a);
  tmp[1] = k;
  p.reinit (tmp);

  tmp[1] = 1;
  tmp[0] = -a;
  Poly l1 (tmp);
  tmp[0] = -b;
  Poly l2 (tmp);
  l1 *= l2;
  
  tmp[1] = 1./6. / (a - b);
  tmp[0] = 1./6. * (a - 2 * b) / (a-b);
  Poly p1 (tmp);
  p1 *= l1;
  p1 *= dda;
  
  tmp[1] *= -1;
  tmp[0] = 1./6. * (b - 2 * a) / (b-a);
  Poly p2 (tmp);
  p2 *= l1;
  p2 *= ddb;
  
  p += p1;
  p += p2;
}  


void Interpolation::secondDerivativeInterpol (
    const std::vector<double >::const_iterator & xbegin,
    const std::vector<double >::const_iterator & xend,
    const std::vector<double >::const_iterator & vbegin,
    const std::vector<double >::const_iterator & ddbegin,
    PiecewisePoly & ps)
{
  ps.clear();
  std::vector<double >::const_iterator xb (xbegin), vb (vbegin), ddb (ddbegin);
  std::vector<double >::const_iterator xp (xbegin), vp (vbegin), ddp (ddbegin);
  ++xp, ++vp, ++ddp;
  while (xp != xend){
    ps.get_x().push_back (*xb);
    Poly tmpp;
    pieceSecondDerivativeInterpol (*(xb++), *(xp++), 
				   *(vb++), *(vp++),
				   *(ddb++), *(ddp++),
				   tmpp);
    ps.get_p().push_back (tmpp);
  }
  ps.get_x().push_back (*xb);
}


void Interpolation::pieceHermiteInterpol (const double & a, const double & b,
					  const double & va, const double & vb,
					  const double & da, const double & db,
					  Poly & p)
{
  std::vector<double > tmp (2,0);
  Poly t ;
  tmp[0] = (-2 * a / (b - a) + 1);
  tmp[1] = (2 / (b - a));
  Poly a0 (tmp);
  tmp[0] = - b / (a - b);
  tmp[1] = 1 / (a - b);
  t.reinit(tmp);
  a0 *= t;
  a0 *= t;
  tmp[0] = - 2 * b / (a - b) + 1;
  tmp[1] = 2 / (a - b);
  Poly a1 (tmp);
  tmp[0] = - a / (b - a);
  tmp[1] = 1 / (b - a);
  t.reinit (tmp);
  a1 *= t;
  a1 *= t;

  tmp[0] = -a;
  tmp[1] = 1;
  Poly b0 (tmp);
  tmp[0] = - b / (a - b);
  tmp[1] = 1 / (a - b);
  t.reinit(tmp);
  b0 *= t;
  b0 *= t;
  tmp[0] = -b;
  tmp[1] = 1;
  Poly b1 (tmp);
  tmp[0] = - a / (b - a);
  tmp[1] = 1 / (b - a);
  t.reinit (tmp);
  b1 *= t;
  b1 *= t;

  p.zero();
  a0 *= va;
  a1 *= vb;
  b0 *= da;
  b1 *= db;
  p += a0;
  p += a1;
  p += b0;
  p += b1;
}

// lbegin--lend, stores lambda
// ubegin--uend, stores mu
bool Interpolation::solverForSplinePeriodic (
    const std::vector<double >::const_iterator & lbegin,
    const std::vector<double >::const_iterator & lend,
    const std::vector<double >::iterator & ubegin, 
    const std::vector<double >::iterator & uend)
{
  std::vector<double > la, lb, lc, ld;
  for (std::vector<double >::const_iterator i = lbegin;
       i != lend; ++i){
    la.push_back (1 - *i);
    lb.push_back (2);
    lc.push_back (*i);
    ld.push_back (0);
  }
//  ld.front() = 1 - *lbegin;
  ld[0] = 1 - lc[0];
  int num = ld.size();
  ld[num-2] = lc[num-2];
  ld[num-1] = lb[num-1];
  
  std::vector<double >::iterator pu = ubegin;
  std::vector<double >::iterator pu_1 = pu ++;
  for (int i = 1; i < num-1; ++i, ++pu, ++pu_1){
    if (lb[i-1] == 0){
      return false;
    }
    double ratio = - la[i] / lb[i-1];
    lb[i] += lc[i-1] * ratio;
    ld[i] += ld[i-1] * ratio;
    *pu += *pu_1 * ratio;
  }
  int i = num-1;
  if (lb[i-1] == 0){
    return false;
  }
  double ratio = - la[i] / lb[i-1];
  lb[i] += ld[i-1] * ratio;
  ld[i] = lb[i];
  *pu += *pu_1 * ratio;
  
//   std::cout << lc.back() << std::endl;
//   std::cout << lc.front() << std::endl;
  ratio = -lb[0] / lc.back();
  ld[0] += ratio * ld[num-1];
  *ubegin += ratio * *pu;
  lb[0] = 0;

//   std::cout << ld.size() << std::endl;
  ld.insert (ld.begin(), ld.back());
//   std::cout << ld.size() << std::endl;
  ld.pop_back();
//   std::cout << ld.size() << std::endl;
  double before = 0.;
//   std::cout << "##############################" << std::endl;
//   std::copy(ubegin, uend, std::ostream_iterator<double >(std::cout, "\n"));
//   std::cout << "##############################" << std::endl;
  for (std::vector<double >::iterator tmpu = ubegin; tmpu != uend; ++tmpu){
    if (tmpu ==  ubegin) {
      before = *tmpu;
      *tmpu = *pu;
    }
    else {
      double beforetmp = *tmpu;
      *tmpu = before;
      before = beforetmp;
    }
  }
//   std::copy(ubegin, uend, std::ostream_iterator<double >(std::cout, "\n"));
//   std::cout << "##############################" << std::endl;
  lc.insert (lc.begin(), *lbegin);
  lc.pop_back ();
  lc.back () = ld.back();
  lb.insert (lb.begin(), 0.);
  lb.pop_back ();
  
  pu = ubegin;
  pu ++;
  pu_1 = pu ++;
  for (int i = 2; i < num-1; ++i, ++pu, ++pu_1){
    if (lc[i-1] == 0){
      return false;
    }
    double ratio = - lb[i] / lc[i-1];
    ld[i] += ld[i-1] * ratio;
    *pu += *pu_1 * ratio;
  }
  i = num-1;
  if (lc[i-1] == 0){
    return false;
  }
  ratio = - lb[i] / lc[i-1];
  lc[i] += ld[i-1] * ratio;
  ld[i] = lc[i];
  *pu += *pu_1 * ratio;


  *pu /=lc[num-1];
  for (int i = num-2; i >= 0; --i, -- pu_1){
    *pu_1 = (*pu_1 - *pu * ld[i]) / lc[i];
  }

  return true;
}

  
  
bool Interpolation::splinePeriodic (const std::vector<double > & x,
				    const std::vector<double > & y,
				    PiecewisePoly & ps)
{
  std::vector<double > lambda (x.size()-1);
  std::vector<double > mu (x.size()-1);
  std::vector<double > dx ;
  
  std::vector<double >::const_iterator i = x.begin();
  std::vector<double >::const_iterator j = i;
  for (++j; j!= x.end(); ++i, ++j){
    dx.push_back(*j - *i);
  }
  lambda[0] = dx.back() / (dx.back() + dx.front());
  mu[0] = 3 * ((1 - lambda.front())/dx.back()*(y[0] - y[y.size()-2]) +
	       lambda.front() / dx.front() * (y[1] - y[0]));
  for (unsigned i = 1; i < lambda.size(); ++i){
    lambda[i] = dx[i-1] / (dx[i-1] + dx[i]);
    mu[i] = 3 * ((1 - lambda[i]) / dx[i-1] * (y[i] - y[i-1]) +
		 lambda[i] / dx[i] * (y[i+1] - y[i]));
  }
  
  bool tag = solverForSplinePeriodic (lambda.begin(), lambda.end(), 
				      mu.begin(), mu.end());
  if (!tag) return false;
  
  ps.get_x() = x;
  ps.get_p().clear();
  for (unsigned i = 0; i < x.size() - 2; ++i){
    Poly tmpp;
    pieceHermiteInterpol (x[i], x[i+1], 
			  y[i], y[i+1], 
			  mu[i], mu[i+1], tmpp);
    ps.get_p().push_back (tmpp);
  }
  Poly tmpp;
  pieceHermiteInterpol (x[x.size()-2], x[x.size()-2+1], 
			y[x.size()-2], y[x.size()-2+1], 
			mu[x.size()-2], mu[0], tmpp);
  ps.get_p().push_back (tmpp);
  return true;
}


bool Interpolation::spline (const std::vector<double > & x,
			    const std::vector<double > & y,
			    PiecewisePoly & ps)
{
  std::vector<double > lambda (x.size());
  std::vector<double > mu (x.size());
  std::vector<double > m (x.size());
  std::vector<double > dx ;
  
  std::vector<double >::const_iterator i = x.begin();
  std::vector<double >::const_iterator j = i;
  for (++j; j!= x.end(); ++i, ++j){
    dx.push_back(*j - *i);
  }
  
  lambda.front() = 1;
  lambda.back() = 0;
  mu.front() = 3 * ((*(++(y.begin()))) - y.front()) / dx.front();
  mu.back()  = 3 * (y.back() - (*(++(y.rbegin())))) / dx.back();
  std::vector<double >::iterator pdx0 = dx.begin();
  std::vector<double >::iterator pdx1 = pdx0;
  ++ pdx1 ;
  std::vector<double >::const_iterator py0 = y.begin();
  std::vector<double >::const_iterator py1 = py0;
  ++ py1;
  std::vector<double >::const_iterator py2 = py1;
  ++ py2;
  std::vector<double >::iterator plambda = lambda.begin();
  ++ plambda;
  std::vector<double >::iterator pmu = mu.begin();
  ++ pmu;
  for (; py2 != y.end(); 
       ++pdx0, ++pdx1, ++py0, ++py1, ++py2, ++plambda, ++pmu){
    *plambda = *pdx0 / (*pdx0 + *pdx1);
    *pmu = 3 * ((1-*plambda) / *pdx0 * (*py1 - *py0) + 
		*plambda / *pdx1 * (*py2 - *py1));
  }

  //   for (unsigned i = 1; i < x.size()-1; ++i){
  //     lambda[i] = dx[i-1] / (dx[i-1] + dx[i]);
  //     mu[i] = 3 * ((1-lambda[i]) / dx[i-1] * (y[i] - y[i-1]) +
  // 		 lambda[i] / dx[i] * (y[i+1] - y[i]));
  //   }
  
  double bet;
  std::vector<double > gam (x.size());
  m[0] = mu[0] / (bet=2);
  for (unsigned j = 1; j < x.size(); ++j){
    gam[j] = lambda[j-1] / bet;
    bet = 2 - (1-lambda[j]) * gam[j];
    if (bet == 0) {
      std::cerr << "a error in triangle solver\n" ;
      return false;
    }
    m[j] = (mu[j] - (1-lambda[j]) * m[j-1]) / bet;
  }
  for (int j = x.size()-2; j >= 0; --j){
    m[j] -= gam[j+1] * m[j+1];
  }

  ps.clear();
  ps.get_x() = x;
  std::vector<double >::const_iterator px0 = x.begin();
  std::vector<double >::const_iterator px1 = px0;
  ++ px1;
  py0 = y.begin();
  py1 = py0;
  ++ py1;
  std::vector<double >::iterator pm0 = m.begin();
  std::vector<double >::iterator pm1 = pm0;
  ++ pm1;
  for (; px1 != x.end(); 
       ++px0, ++px1, ++py0, ++py1, ++pm0, ++pm1){
    Poly tmpp;
    pieceHermiteInterpol (*px0, *px1, *py0, *py1, *pm0, *pm1, tmpp);
    ps.get_p().push_back(tmpp);
  }

  return true;
}


bool Interpolation::spline (const std::vector<double >::const_iterator xbegin,
			    const std::vector<double >::const_iterator xend,
			    const std::vector<double >::const_iterator ybegin,
			    PiecewisePoly & ps)
{
  int xsize = 0;
  std::vector<double >::const_iterator itmp = xbegin;
  while (itmp ++ != xend) ++ xsize;
  
  std::vector<double > lambda (xsize);
  std::vector<double > mu (xsize);
  std::vector<double > m (xsize);
  std::vector<double > dx ;

  // setup linear system
  std::vector<double >::const_iterator i = xbegin;
  std::vector<double >::const_iterator j = i;
  for (++j; j!= xend; ++i, ++j){
    dx.push_back(*j - *i);
  }
  lambda.front() = 1;
  lambda.back() = 0;
  mu.front() = 3 * ((*(++(itmp = ybegin))) - *ybegin) / dx.front();
  std::vector<double >::iterator pdx0 = dx.begin();
  std::vector<double >::iterator pdx1 = pdx0;
  ++ pdx1 ;
  std::vector<double >::const_iterator py0 = ybegin;
  std::vector<double >::const_iterator py1 = py0;
  ++ py1;
  std::vector<double >::const_iterator py2 = py1;
  ++ py2;
  std::vector<double >::iterator plambda = lambda.begin();
  ++ plambda;
  std::vector<double >::iterator pmu = mu.begin();
  ++ pmu;
  for (; pdx1 != dx.end(); 
       ++pdx0, ++pdx1, ++py0, ++py1, ++py2, ++plambda, ++pmu){
    *plambda = *pdx0 / (*pdx0 + *pdx1);
    *pmu = 3 * ((1-*plambda) / *pdx0 * (*py1 - *py0) + 
		*plambda / *pdx1 * (*py2 - *py1));
  }
  mu.back()  = 3 * (*py1 - *py0) / dx.back();
  
  // solve tridiangonal linear system
  double bet;
  std::vector<double > gam (xsize);
  m[0] = mu[0] / (bet=2);
  for (int j = 1; j < xsize; ++j){
    gam[j] = lambda[j-1] / bet;
    bet = 2 - (1-lambda[j]) * gam[j];
    if (bet == 0) {
      std::cerr << "a error in triangle solver\n" ;
      return false;
    }
    m[j] = (mu[j] - (1-lambda[j]) * m[j-1]) / bet;
  }
  for (int j = xsize-2; j >= 0; --j){
    m[j] -= gam[j+1] * m[j+1];
  }

  // make piecewise polynominal
  ps.get_p().clear();
  ps.get_x().resize(xsize);
  std::copy (xbegin, xend, ps.get_x().begin());
  std::vector<double >::const_iterator px0 = xbegin;
  std::vector<double >::const_iterator px1 = px0;
  ++ px1;
  py0 = ybegin;
  py1 = py0;
  ++ py1;
  std::vector<double >::iterator pm0 = m.begin();
  std::vector<double >::iterator pm1 = pm0;
  ++ pm1;
  for (; px1 != xend; 
       ++px0, ++px1, ++py0, ++py1, ++pm0, ++pm1){
    Poly tmpp;
    pieceHermiteInterpol (*px0, *px1, *py0, *py1, *pm0, *pm1, tmpp);
    ps.get_p().push_back(tmpp);
  }

  return true;
}



// void tridag(float a[], float b[], float c[], float r[], float u[],
// 	    unsigned long n)
// //Solves for a vector u[1..n] the tridiagonal linear set given by equation (2.4.1). a[1..n],
// //  b[1..n], c[1..n], and r[1..n] are input vectors and are not modified.
// {
//   unsigned long j;
//   float bet,*gam;
//   gam=vector(1,n); //One vector of workspace, gam is needed.
//   //If this happens then you should rewrite your equations as a set of order N-1, w ith u2
//   //trivially eliminated.
//   u[0]=r[0]/(bet=2);
//   for (j=1;j<=n;j++) { //Decomposition and forward substitution.
//     gam[j]=c[j-1]/bet;
//     bet=2-a[j]*gam[j];
//     if (bet == 0.0) //nrerror("Error 2 in tridag"); //Algorithm fails; see be
//     u[j]=(r[j]-a[j]*u[j-1])/bet; //low.
//   }
//   for (j=(n-1);j>=1;j--)
//     u[j] -= gam[j+1]*u[j+1]; //Backsubstitution.
//   free_vector(gam,1,n);
// }
