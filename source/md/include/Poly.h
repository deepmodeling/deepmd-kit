#ifndef __wanghan_Poly_h__
#define __wanghan_Poly_h__

#include <iostream>
#include <string>
#include <vector>

class Poly
{
  std::vector<double > a;
  unsigned order;
public:
  Poly ();
  Poly (const std::vector<double > & out);
  void reinit (const std::vector<double > & out);
  void zero () {a.clear(); a.resize(1,0); order = 0;}
  void one  () {a.clear(); a.resize(1,1); order = 0;}
public:
  Poly & operator = (const Poly & poly);
  Poly & operator += (const Poly & poly);
  Poly & operator += (const double & b);
  Poly & operator *= (const Poly & poly);
  Poly & operator *= (const double & scale);
  Poly & derivative ();
public:
  unsigned & getOrder () {return order;}
  const unsigned & getOrder () const {return order;}
  std::vector<double > & getCoeffs () {return a;}
  const std::vector<double > & getCoeffs () const {return a;}
public:
  void print ();
  void print (const std::string & x);
  void printCode (const std::string & x);
public :
  double value ( const double & x ) const;
public:
  // p = f(ax + b)
  Poly & valueLinearPoly (const double & a, const double & b,
			  Poly & p);
}
    ;

class PiecewisePoly
{
public:
  std::vector<double > & get_x () {return x;}
  std::vector<Poly   > & get_p () {return p;}
  const std::vector<double > & get_x () const {return x;}
  const std::vector<Poly   > & get_p () const {return p;}
public:
  void clear () {x.clear(); p.clear();}
  bool valid () const;
public:
  double value (const double & r) const;
  void   value (const std::vector<double > & r,
		std::vector<double > & y) const;
  double value_periodic (const double & r) const;
  void   value_periodic (const std::vector<double > & r,
			 std::vector<double > & y) const;
private:
  std::vector<double > x;
  std::vector<Poly > p;
  void value (const unsigned & xbegin,
	      const unsigned & xend,
	      const std::vector<double > & r,
	      const unsigned & rbegin,
	      const unsigned & rend,
	      std::vector<double > & y) const;
  double value (const double & xx,
		unsigned & begin,
		unsigned & end) const;  
}
    ;



#endif
