// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

namespace deepmd {

template <typename FPTYPE>
struct Region {
  FPTYPE* boxt;
  FPTYPE* rec_boxt;
  Region();
  Region(FPTYPE* extern_boxt, FPTYPE* extern_rec_boxt);
  ~Region();

 private:
  bool self_allocated;
};

template <typename FPTYPE>
void init_region_cpu(Region<FPTYPE>& region, const FPTYPE* boxt);

template <typename FPTYPE>
FPTYPE volume_cpu(const Region<FPTYPE>& region);

template <typename FPTYPE>
void convert_to_inter_cpu(FPTYPE* ri,
                          const Region<FPTYPE>& region,
                          const FPTYPE* rp);

template <typename FPTYPE>
void convert_to_phys_cpu(FPTYPE* rp,
                         const Region<FPTYPE>& region,
                         const FPTYPE* ri);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// only for unittest
template <typename FPTYPE>
void convert_to_inter_gpu(FPTYPE* ri,
                          const Region<FPTYPE>& region,
                          const FPTYPE* rp);

template <typename FPTYPE>
void convert_to_phys_gpu(FPTYPE* rp,
                         const Region<FPTYPE>& region,
                         const FPTYPE* ri);

template <typename FPTYPE>
void volume_gpu(FPTYPE* volume, const Region<FPTYPE>& region);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
