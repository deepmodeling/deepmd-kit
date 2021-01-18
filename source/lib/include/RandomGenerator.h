#ifndef __wanghan_ToolBox_h__
#define __wanghan_ToolBox_h__
#include<vector>

typedef double value_type;

namespace RandomGenerator_MT19937{
    void init_by_array(unsigned long init_key[], int key_length);
    void init_genrand(unsigned long s);
    unsigned long genrand_int32(void);
    long genrand_int31(void);
    double genrand_real1(void); // in [0,1]
    double genrand_real2(void); // in [0,1)
    double genrand_real3(void); // in (0,1)
    double genrand_res53(void);
  double gaussian ();		// Box Muller
  void sphere (double & x, double & y, double & z);
}




#endif
