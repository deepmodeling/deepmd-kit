#ifndef __Stopwatch_h_wanghan__
#define __Stopwatch_h_wanghan__

#include <sys/param.h>
#include <sys/times.h>
#include <sys/types.h>

class Stopwatch 
{ 
public: 
  Stopwatch(): HZi (1./HZ) {}; 
 
  void              start(); 
  void              stop(); 
  double            system() const; 
  double            user() const; 
  double            real() const; 
 
  static double     resolution() {return 1./HZ;}; 
private:
  struct tms tic, toc;
  long r1, r0;
  double HZi;
};               

inline double Stopwatch::user () const
{
  return (double)(toc.tms_utime - tic.tms_utime) * HZi;
}

inline double Stopwatch::system () const
{
  return (double)(toc.tms_stime - tic.tms_stime) * HZi;
}

inline double Stopwatch::real () const
{
  return (double)(r1 - r0) * HZi;
}

inline void Stopwatch::stop ()
{
  r1 = times (&toc);
}

inline void Stopwatch::start() 
{
  r0 = times (&tic);
}


#endif
// end of file 

