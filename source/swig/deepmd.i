%module deepmd

%{
#include "deepmd.hpp"
using namespace deepmd::hpp;
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_except.i"
%include "typemaps.i"

%exception {
  try {
    $action
  } catch (const deepmd::hpp::deepmd_exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
};

%apply double& INOUT { double& ener };

%include "deepmd.hpp"

%extend deepmd::hpp::DeepPot {
   %template(compute) compute<double,double>;
   %template(compute_mixed_type) compute_mixed_type<double,double>;
};
%extend deepmd::hpp::DeepPotModelDevi {
   %template(compute) compute<double>;
};
%extend deepmd::hpp::DeepTensor {
   %template(compute) compute<double>;
};
%extend deepmd::hpp::DipoleChargeModifier {
   %template(compute) compute<double>;
};
