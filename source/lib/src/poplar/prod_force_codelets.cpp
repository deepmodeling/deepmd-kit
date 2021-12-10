// Copyright 2021 Graphcore Ltd.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <ipu_memory_intrinsics>
#include <print.h>

using namespace poplar;

static constexpr auto SPAN    = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template<typename T>
struct FloatDef{
};

template<>
struct FloatDef<float>{
  static inline const int         kSftBits  = 1;
  static inline const int         kStep     = (1 << kSftBits);
  static inline const int         kPlus     = kStep - 1;
  static inline constexpr float2  kZeroV    = { 0.0f, 0.0f };
  typedef   float2                FVType;
  static inline float             sumV(FVType const& v)
  {
    return v[0] + v[1];
  };
  static inline void              pack(FVType const& a, FVType const& b, FVType const& c, FVType& x, FVType& y, FVType& z)
  {
    x = { a[0], b[1]} ;
    y = { a[1], c[0]} ;
    z = { b[0], c[1]} ;
  };

  static inline void              pack(FVType const& a, FVType const& b, FVType const& c, FVType const& d, FVType const& e, FVType const& f, 
                                       FVType& v0, FVType& v1, FVType& v2, FVType& v3, FVType& v4, FVType& v5)
  {
    v0 = { a[0], b[1] };
    v1 = { d[0], e[1] };
    v2 = { a[1], c[0] };
    v3 = { d[1], f[0] };
    v4 = { b[0], c[1] };
    v5 = { e[0], f[1] };
  };  
};

template<>
struct FloatDef<half>{
  static inline const int         kSftBits  = 2;
  static inline const int         kStep     = (1 << kSftBits);
  static inline const int         kPlus     = kStep - 1;
  static inline constexpr half4   kZeroV    = { 0.0f, 0.0f, 0.0f, 0.0f };
  typedef   half4                 FVType;
  static inline half              sumV(FVType const& v)
  {
    return v[0] + v[1] + v[2] + v[3];
  };
  static inline void              pack(FVType const& a, FVType const& b, FVType const& c, FVType& x, FVType& y, FVType& z)
  {
    x = { a[0], a[3], b[2], c[1] } ;
    y = { a[1], b[0], b[3], c[2] } ;
    z = { a[2], b[1], c[0], c[3] } ;
  };
  static inline void              pack(FVType const& a, FVType const& b, FVType const& c, FVType const& d, FVType const& e, FVType const& f, 
                                       FVType& v0, FVType& v1, FVType& v2, FVType& v3, FVType& v4, FVType& v5)
  {
    v0 = { a[0], a[3], b[2], c[1] };
    v1 = { d[0], d[3], e[2], f[1] };
    v2 = { a[1], b[0], b[3], c[2] };
    v3 = { d[1], e[0], e[3], f[2] };
    v4 = { a[2], b[1], c[0], c[3] };
    v5 = { d[2], e[1], f[0], f[3] };
  }; 
};

template <typename T>
class ProdForceFirstVertex : public Vertex {
public:
  ProdForceFirstVertex();
  Input<Vector<T, ONE_PTR, 8>>       net_deriv_;
  Input<Vector<T, ONE_PTR, 8>>       env_deriv_;
  InOut<Vector<T, SPAN, 8>>          force_;

  const int                          nnei_;

  template<typename FType, typename FVType>
  static void prod_force(FVType const*  net_deriv,
                         FVType const*  env_deriv,
                         int            loop,
                         int            desc_cnt,
                         FType*         force)
  {
    FVType cur_env_deriv_x, cur_env_deriv_y, cur_env_deriv_z;
    for(int i = 0 ; i < loop ; i ++)
    {
      FVType  x = FloatDef<FType>::kZeroV;
      FVType  y = FloatDef<FType>::kZeroV;
      FVType  z = FloatDef<FType>::kZeroV;
      for(int j = 0 ; j < desc_cnt ; j ++)
      {
        FVType cur_net_deriv    = ipu::load_postinc(&net_deriv, 1);
        FVType cur_env_deriv_a  = ipu::load_postinc(&env_deriv, 1);
        FVType cur_env_deriv_b  = ipu::load_postinc(&env_deriv, 1);
        FVType cur_env_deriv_c  = ipu::load_postinc(&env_deriv, 1);
        FloatDef<FType>::pack(cur_env_deriv_a, cur_env_deriv_b, cur_env_deriv_c, cur_env_deriv_x, cur_env_deriv_y, cur_env_deriv_z);
        x                      -= cur_net_deriv * cur_env_deriv_x;
        y                      -= cur_net_deriv * cur_env_deriv_y;
        z                      -= cur_net_deriv * cur_env_deriv_z;
      }
      force[3 * i]     += FloatDef<FType>::sumV(x);
      force[3 * i + 1] += FloatDef<FType>::sumV(y);
      force[3 * i + 2] += FloatDef<FType>::sumV(z);
    }
  }

  bool compute() {
    int                                  data_cnt      = force_.size();
    int                                  desc_cnt      = (nnei_ << 2);
    int                                  loop          = data_cnt / 3;
    typename FloatDef<T>::FVType const*  net_deriv_ptr = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(net_deriv_[0]));
    typename FloatDef<T>::FVType const*  env_deriv_ptr = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(env_deriv_[0]));
    T*                                   force_ptr     = reinterpret_cast<T*>(&(force_[0]));
    desc_cnt = desc_cnt >> FloatDef<T>::kSftBits;
    prod_force<T, typename FloatDef<T>::FVType>(net_deriv_ptr, 
                                                env_deriv_ptr, 
                                                loop, 
                                                desc_cnt, 
                                                force_ptr);
    return true;
  }
};


template <typename T>
class ProdForceSecondVertex : public Vertex {
public:
  ProdForceSecondVertex();
  Input<Vector<T, SPAN, 8>>          net_deriv_;
  Input<Vector<T, ONE_PTR, 8>>       env_deriv_;
  Input<Vector<int, ONE_PTR>>        nlist_;
  InOut<Vector<T, SPAN, 8>>          force_;

  const int                          nloc_; 
  const int                          nnei_;
  const int                          start_;

  template<typename FType, typename FVType>
  static void prod_force(FVType const*  net_deriv,
                         FVType const*  env_deriv,
                         int const*     nlist,
                         int            loop,
                         int            nnei,
                         int            start,
                         int            end,
                         FType*         force)
  {
    int desc_cnt = (nnei << 2);
    desc_cnt     = desc_cnt >> FloatDef<T>::kSftBits;
    FVType const*  cur_net_deriv_ptr = net_deriv;
    FVType const*  cur_env_deriv_ptr = env_deriv;
    int const*     cur_nlist_ptr     = nlist;
    FType*         cur_force_ptr     = force;

    FVType  cur_env_derive_g, cur_env_derive_h, cur_env_derive_i ;
    FVType  cur_env_derive_j, cur_env_derive_k, cur_env_derive_l ;
    for(int i = 0 ; i < loop ; i ++)
    {
      for (int j = 0; j < nnei; ++j) 
      {
        int j_idx = cur_nlist_ptr[j];
        if (j_idx < 0 || j_idx < start || j_idx >= end) continue;

        j_idx   = j_idx - start;
        int     desc_pos         = (j << 2) >> FloatDef<FType>::kSftBits;
        FVType  x, y, z;
        int     src_pos          = desc_pos;
        FVType  cur_net_derive_a = cur_net_deriv_ptr[src_pos];
        FVType  cur_net_derive_b = cur_net_deriv_ptr[src_pos + 1];
        FVType  cur_env_derive_a = cur_env_deriv_ptr[3 * src_pos];
        FVType  cur_env_derive_b = cur_env_deriv_ptr[3 * src_pos + 1];
        FVType  cur_env_derive_c = cur_env_deriv_ptr[3 * src_pos + 2];
        FVType  cur_env_derive_d = cur_env_deriv_ptr[3 * src_pos + 3];
        FVType  cur_env_derive_e = cur_env_deriv_ptr[3 * src_pos + 4];
        FVType  cur_env_derive_f = cur_env_deriv_ptr[3 * src_pos + 5];
        FloatDef<FType>::pack(cur_env_derive_a, cur_env_derive_b, cur_env_derive_c, 
                              cur_env_derive_d, cur_env_derive_e, cur_env_derive_f,
                              cur_env_derive_g, cur_env_derive_h, cur_env_derive_i, 
                              cur_env_derive_j, cur_env_derive_k, cur_env_derive_l);

        x  = cur_net_derive_a * cur_env_derive_g;
        y  = cur_net_derive_a * cur_env_derive_i;
        z  = cur_net_derive_a * cur_env_derive_k;

        x += cur_net_derive_b * cur_env_derive_h;
        force[j_idx * 3]     += FloatDef<FType>::sumV(x);
        y += cur_net_derive_b * cur_env_derive_j;
        force[j_idx * 3 + 1] += FloatDef<FType>::sumV(y);
        z += cur_net_derive_b * cur_env_derive_l;
        force[j_idx * 3 + 2] += FloatDef<FType>::sumV(z);
      }
      cur_net_deriv_ptr += desc_cnt;
      cur_env_deriv_ptr += 3 * desc_cnt;
      cur_nlist_ptr     += nnei;
    }
  }

  bool compute() {
    int                                  data_cnt      = net_deriv_.size();
    int                                  loop          = (data_cnt >> 2) / nnei_;
    typename FloatDef<T>::FVType const*  net_deriv_ptr = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(net_deriv_[0]));
    typename FloatDef<T>::FVType const*  env_deriv_ptr = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(env_deriv_[0]));
    int const*                           nlist_ptr     = reinterpret_cast<int const*>(&(nlist_[0]));
    T*                                   force_ptr     = reinterpret_cast<T*>(&(force_[0]));
    int                                  output_len    = force_.size() / 3;
    prod_force<T, typename FloatDef<T>::FVType>(net_deriv_ptr, 
                                                env_deriv_ptr, 
                                                nlist_ptr,
                                                loop, 
                                                nnei_,
                                                start_,
                                                start_ + output_len,
                                                force_ptr);
    return true;
  }
};

template class ProdForceFirstVertex<float>;
template class ProdForceFirstVertex<half>;
template class ProdForceSecondVertex<float>;
template class ProdForceSecondVertex<half>;
