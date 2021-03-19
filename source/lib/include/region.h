#pragma once

namespace deepmd{

template<typename FPTYPE>
struct Region
{
  FPTYPE * boxt;
  FPTYPE * rec_boxt;
  Region();
  ~Region();
};

template<typename FPTYPE>
void
init_region_cpu(
    Region<FPTYPE> & region,
    const FPTYPE * boxt);

template<typename FPTYPE>
FPTYPE
volume_cpu(
    const Region<FPTYPE> & region);

template<typename FPTYPE>
void
convert_to_inter_cpu(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp);

template<typename FPTYPE>
void
convert_to_phys_cpu(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri);

}

