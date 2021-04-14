#include "utilities.h"

// functions used in custom ops
void deepmd::cum_sum(
    std::vector<int> & sec, 
    const std::vector<int> & n_sel) 
{
  sec.resize (n_sel.size() + 1);
  sec[0] = 0;
  for (int ii = 1; ii < sec.size(); ++ii) {
    sec[ii] = sec[ii-1] + n_sel[ii-1];
  }
}
