#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <ipu_memory_intrinsics>

using namespace poplar;

static constexpr auto SPAN    = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template<typename T>
struct FloatDef{
};

template<>
struct FloatDef<float>{
  static inline const int    kSftBits  = 1;
  static inline const int    kStep     = (1 << kSftBits);
  static inline const int    kPlus     = kStep - 1;
  typedef   float2           FVType;
};

template<>
struct FloatDef<half>{
  static inline const int    kSftBits  = 2;
  static inline const int    kStep     = (1 << kSftBits);
  static inline const int    kPlus     = kStep - 1;
  typedef   half4            FVType;
};

template <typename T>
class ProdVirialVertex : public Vertex {
public:
  ProdVirialVertex();
  Input<Vector<T, SPAN, 8>>          net_deriv_;
  Input<Vector<T, ONE_PTR, 8>>       env_deriv_;
  Input<Vector<T, ONE_PTR, 8>>       rij_;
  Input<Vector<int, ONE_PTR>>        nlist_;
  InOut<Vector<T, ONE_PTR, 8>>       virial_;
  InOut<Vector<T, SPAN, 8>>          atom_virial_;

  const int                          nloc_; 
  const int                          nnei_;
  const int                          start_;

  template<typename FType, typename FVType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static inline void proc(FVType const*   net_deriv,
                          FVType const*   env_deriv,
                          FType           rij_0,
                          FType           rij_1,
                          FType           rij_2,
                          int             desc_pos,
                          FType*          virial,
                          FType*          atom_virial,
                          int             j_idx)
  {
    FVType cur_pref_0 = -net_deriv[desc_pos];
    FVType cur_pref_1 = -net_deriv[desc_pos + 1];

    FVType cur_env_deriv_0 = env_deriv[3 * desc_pos];
    FVType cur_env_deriv_1 = env_deriv[3 * desc_pos + 1];
    FVType cur_env_deriv_2 = env_deriv[3 * desc_pos + 2];
    FVType cur_env_deriv_3 = env_deriv[3 * desc_pos + 3];
    FVType cur_env_deriv_4 = env_deriv[3 * desc_pos + 4];
    FVType cur_env_deriv_5 = env_deriv[3 * desc_pos + 5];

    FVType pref01_rij_0    = cur_pref_0 * rij_0;
    FVType pref01_rij_1    = cur_pref_0 * rij_1;
    FVType pref01_rij_2    = cur_pref_0 * rij_2;
    FVType pref23_rij_0    = cur_pref_1 * rij_0;
    FVType pref23_rij_1    = cur_pref_1 * rij_1;
    FVType pref23_rij_2    = cur_pref_1 * rij_2;

    FVType a_d             = pref01_rij_0[0] * cur_env_deriv_0;
    FVType g_a             = (FVType){pref01_rij_0[0], pref01_rij_0[1]} * cur_env_deriv_1;
    FVType d_g             = pref01_rij_0[1] * cur_env_deriv_2;
    FVType b_e             = pref01_rij_1[0] * cur_env_deriv_0;
    FVType h_b             = (FVType){pref01_rij_1[0], pref01_rij_1[1]} * cur_env_deriv_1;
    FVType e_h             = pref01_rij_1[1] * cur_env_deriv_2;
    FVType c_f             = pref01_rij_2[0] * cur_env_deriv_0;
    FVType i_c             = (FVType){pref01_rij_2[0], pref01_rij_2[1]} * cur_env_deriv_1;
    FVType f_i             = pref01_rij_2[1] * cur_env_deriv_2;

    a_d += pref23_rij_0[0] * cur_env_deriv_3;
    g_a += (FVType){pref23_rij_0[0], pref23_rij_0[1]} * cur_env_deriv_4;
    d_g += pref23_rij_0[1] * cur_env_deriv_5;
    b_e += pref23_rij_1[0] * cur_env_deriv_3;
    h_b += (FVType){pref23_rij_1[0], pref23_rij_1[1]} * cur_env_deriv_4;
    e_h += pref23_rij_1[1] * cur_env_deriv_5;
    c_f += pref23_rij_2[0] * cur_env_deriv_3;
    i_c += (FVType){pref23_rij_2[0], pref23_rij_2[1]} * cur_env_deriv_4;
    f_i += pref23_rij_2[1] * cur_env_deriv_5;

    FType   i  = i_c[0] + f_i[1];

    FVType  x0 = { a_d[0], b_e[0] };
    FVType  y0 = { c_f[0], a_d[1] };
    FVType  z0 = { b_e[1], c_f[1] };
    FVType  w0 = { g_a[0], h_b[0] };
    FVType  x1 = { g_a[1], h_b[1] };
    FVType  y1 = { i_c[1], d_g[0] };
    FVType  z1 = { e_h[0], f_i[0] };
    FVType  w1 = { d_g[1], e_h[1] };

    x0 = x0 + x1;
    y0 = y0 + y1;
    z0 = z0 + z1;
    w0 = w0 + w1;

    ((FVType*)virial)[0] -= x0;
    ((FVType*)virial)[1] -= y0;
    ((FVType*)virial)[2] -= z0;
    ((FVType*)virial)[3] -= w0;
    virial[8]            -= i;
    atom_virial[9 * j_idx]     -= x0[0];
    atom_virial[9 * j_idx + 1] -= x0[1];
    atom_virial[9 * j_idx + 2] -= y0[0];
    atom_virial[9 * j_idx + 3] -= y0[1];
    atom_virial[9 * j_idx + 4] -= z0[0];
    atom_virial[9 * j_idx + 5] -= z0[1];
    atom_virial[9 * j_idx + 6] -= w0[0];
    atom_virial[9 * j_idx + 7] -= w0[1];
    atom_virial[9 * j_idx + 8] -= i;
  }

  template<typename FType, typename FVType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static inline void proc(FVType const*   net_deriv,
                          FVType const*   env_deriv,
                          FType           rij_0,
                          FType           rij_1,
                          FType           rij_2,
                          int             desc_pos,
                          FType*          virial,
                          FType*          atom_virial,
                          int             j_idx)
  {
    FVType cur_pref        = -net_deriv[desc_pos];

    FVType cur_env_deriv_0 = env_deriv[3 * desc_pos];
    FVType cur_env_deriv_1 = env_deriv[3 * desc_pos + 1];
    FVType cur_env_deriv_2 = env_deriv[3 * desc_pos + 2];

    FVType pref_rij_0      = cur_pref * rij_0;
    FVType pref_rij_1      = cur_pref * rij_1;
    FVType pref_rij_2      = cur_pref * rij_2;

    FVType a_d_g_a  = (FVType){pref_rij_0[0], pref_rij_0[0], pref_rij_0[0], pref_rij_0[1]} * cur_env_deriv_0;
    FVType d_g_a_d  = (FVType){pref_rij_0[1], pref_rij_0[1], pref_rij_0[2], pref_rij_0[2]} * cur_env_deriv_1;
    FVType g_a_d_g  = (FVType){pref_rij_0[2], pref_rij_0[3], pref_rij_0[3], pref_rij_0[3]} * cur_env_deriv_2;

    FVType b_e_h_b  = (FVType){pref_rij_1[0], pref_rij_1[0], pref_rij_1[0], pref_rij_1[1]} * cur_env_deriv_0;
    FVType e_h_b_e  = (FVType){pref_rij_1[1], pref_rij_1[1], pref_rij_1[2], pref_rij_1[2]} * cur_env_deriv_1;
    FVType h_b_e_h  = (FVType){pref_rij_1[2], pref_rij_1[3], pref_rij_1[3], pref_rij_1[3]} * cur_env_deriv_2;

    FVType c_f_i_c  = (FVType){pref_rij_2[0], pref_rij_2[0], pref_rij_2[0], pref_rij_2[1]} * cur_env_deriv_0;
    FVType f_i_c_f  = (FVType){pref_rij_2[1], pref_rij_2[1], pref_rij_2[2], pref_rij_2[2]} * cur_env_deriv_1;
    FVType i_c_f_i  = (FVType){pref_rij_2[2], pref_rij_2[3], pref_rij_2[3], pref_rij_2[3]} * cur_env_deriv_2;
/*
    a = a_d_g_a[0] + a_d_g_a[3] + d_g_a_d[2] + g_a_d_g[1];
    b = b_e_h_b[0] + b_e_h_b[3] + e_h_b_e[2] + h_b_e_h[1];
    c = c_f_i_c[0] + c_f_i_c[3] + f_i_c_f[2] + i_c_f_i[1];
    d = a_d_g_a[1] + d_g_a_d[0] + d_g_a_d[3] + g_a_d_g[2];
    e = b_e_h_b[1] + e_h_b_e[0] + e_h_b_e[3] + h_b_e_h[2];
    f = c_f_i_c[1] + f_i_c_f[0] + f_i_c_f[3] + i_c_f_i[2];
    g = a_d_g_a[2] + d_g_a_d[1] + g_a_d_g[0] + g_a_d_g[3];
    h = b_e_h_b[2] + e_h_b_e[1] + h_b_e_h[0] + h_b_e_h[3];
*/
    FType   i  = c_f_i_c[2] + f_i_c_f[1] + i_c_f_i[0] + i_c_f_i[3];

    FVType  x0 = { a_d_g_a[0], b_e_h_b[0], c_f_i_c[0], a_d_g_a[1] };
    FVType  y0 = { b_e_h_b[1], c_f_i_c[1], a_d_g_a[2], b_e_h_b[2] };
    FVType  x1 = { a_d_g_a[3], b_e_h_b[3], c_f_i_c[3], d_g_a_d[0] };
    FVType  y1 = { e_h_b_e[0], f_i_c_f[0], d_g_a_d[1], e_h_b_e[1] };
    FVType  x2 = { d_g_a_d[2], e_h_b_e[2], f_i_c_f[2], d_g_a_d[3] };
    FVType  y2 = { e_h_b_e[3], f_i_c_f[3], g_a_d_g[0], h_b_e_h[0] };
    FVType  x3 = { g_a_d_g[1], h_b_e_h[1], i_c_f_i[1], g_a_d_g[2] };
    FVType  y3 = { h_b_e_h[2], i_c_f_i[2], g_a_d_g[3], h_b_e_h[3] };

    x0 = x0 + x1;
    y0 = y0 + y1;
    x2 = x2 + x3;
    y2 = y2 + y3;
    x0 = x0 + x2;
    y0 = y0 + y2;

    ((FVType*)virial)[0] -= x0;
    ((FVType*)virial)[1] -= y0;
    virial[8]            -= i;
    
    atom_virial[9 * j_idx]     -= x0[0];
    atom_virial[9 * j_idx + 1] -= x0[1];
    atom_virial[9 * j_idx + 2] -= x0[2];
    atom_virial[9 * j_idx + 3] -= x0[3];
    atom_virial[9 * j_idx + 4] -= y0[0];
    atom_virial[9 * j_idx + 5] -= y0[1];
    atom_virial[9 * j_idx + 6] -= y0[2];
    atom_virial[9 * j_idx + 7] -= y0[3];
    atom_virial[9 * j_idx + 8] -= i;
  }

  template<typename FType, typename FVType>
  static void prod_virial(FVType const*   net_deriv,
                          FVType const*   env_deriv,
                          FType const*    rij,
                          int const*      nlist,
                          int             loop,
                          int             nnei,
                          int             start,
                          int             end,
                          FType*          virial,
                          FType*          atom_virial)
  {
    int desc_cnt = (nnei << 2);
    desc_cnt     = desc_cnt >> FloatDef<T>::kSftBits;
    FVType const*  cur_net_deriv_ptr   = net_deriv;
    FVType const*  cur_env_deriv_ptr   = env_deriv;
    FType  const*  cur_rij_ptr         = rij;
    int const*     cur_nlist_ptr       = nlist;
    for(int i = 0 ; i < loop ; ++i)
    {
      for (int j = 0; j < nnei; ++j)
      {
        int j_idx = cur_nlist_ptr[j];
        if ((j_idx < 0) || (j_idx < start) || (j_idx >= end)) continue;

        j_idx  = j_idx - start;

        int    desc_pos   = (j << 2) >> FloatDef<T>::kSftBits;
        FType  rij_0      = cur_rij_ptr[3 * j];
        FType  rij_1      = cur_rij_ptr[3 * j + 1];
        FType  rij_2      = cur_rij_ptr[3 * j + 2];
        FType a, b, c, d, e, f, g, h, m;
        proc<FType, FVType>(cur_net_deriv_ptr,
                            cur_env_deriv_ptr,
                            rij_0,
                            rij_1,
                            rij_2,
                            desc_pos,
                            virial, atom_virial, j_idx);
/*
        virial[0] -= a;
        virial[1] -= b;
        virial[2] -= c;
        virial[3] -= d;
        virial[4] -= e;
        virial[5] -= f;
        virial[6] -= g;
        virial[7] -= h;
        virial[8] -= m;
        atom_virial[9 * j_idx]     -= a;
        atom_virial[9 * j_idx + 1] -= b;
        atom_virial[9 * j_idx + 2] -= c;
        atom_virial[9 * j_idx + 3] -= d;
        atom_virial[9 * j_idx + 4] -= e;
        atom_virial[9 * j_idx + 5] -= f;
        atom_virial[9 * j_idx + 6] -= g;
        atom_virial[9 * j_idx + 7] -= h;
        atom_virial[9 * j_idx + 8] -= m;
*/
      }
      cur_net_deriv_ptr += desc_cnt;
      cur_env_deriv_ptr += 3 * desc_cnt;
      cur_rij_ptr       += 3 * nnei;
      cur_nlist_ptr     += nnei;
    }
  };

  bool compute() {
    int                                  data_cnt        = net_deriv_.size();
    int                                  loop            = (data_cnt >> 2) / nnei_;
    typename FloatDef<T>::FVType const*  net_deriv_ptr   = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(net_deriv_[0]));
    typename FloatDef<T>::FVType const*  env_deriv_ptr   = reinterpret_cast<typename FloatDef<T>::FVType const*>(&(env_deriv_[0]));
    T const*                             rij_ptr         = reinterpret_cast<T const*>(&(rij_[0]));
    int const*                           nlist_ptr       = reinterpret_cast<int const*>(&(nlist_[0]));
    T*                                   virial_ptr      = reinterpret_cast<T*>(&(virial_[0]));
    T*                                   atom_virial_ptr = reinterpret_cast<T*>(&(atom_virial_[0]));
    int                                  output_len      = atom_virial_.size() / 9;
    prod_virial<T, typename FloatDef<T>::FVType>(net_deriv_ptr, 
                                                 env_deriv_ptr, 
                                                 rij_ptr,
                                                 nlist_ptr,
                                                 loop, 
                                                 nnei_,
                                                 start_,
                                                 start_ + output_len,
                                                 virial_ptr,
                                                 atom_virial_ptr);
    return true;
  }
};

template class ProdVirialVertex<float>;
template class ProdVirialVertex<half>;
