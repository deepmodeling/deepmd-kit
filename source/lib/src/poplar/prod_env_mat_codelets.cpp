// Copyright 2021 Graphcore Ltd.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
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
  static inline FVType            setV(float const& a)
  {
    FVType x = { a, a };
    return x;
  }; 
};

template<>
struct FloatDef<half>{
  static inline const int         kSftBits  = 2;
  static inline const int         kStep     = (1 << kSftBits);
  static inline const int         kPlus     = kStep - 1;
  static inline constexpr half4   kZeroV    = { 0.0f, 0.0f, 0.0f, 0.0f };
  typedef   half4                 FVType;
  static inline FVType            setV(half const& a)
  {
    FVType x = { a, a, a, a };
    return x;
  };
};

template <typename T>
class GenerateDistVertex : public Vertex {
public:
  GenerateDistVertex();
  Input<Vector<int, ONE_PTR, 4>>            nlist_;
  Input<Vector<int, ONE_PTR, 4>>            nnumneigh_size_;
  Input<Vector<T,   ONE_PTR, 8>>            npos_;
  Input<Vector<int, ONE_PTR, 4>>            ntype_;
  Input<Vector<T, ONE_PTR, 4>>              pos_loc_;
  Output<Vector<T,  ONE_PTR, 8>>            rij_;
  Output<Vector<float, ONE_PTR, 16, true>>  sel_idx_buf_;
  Output<Vector<int,SPAN, 4>>               fmt_idx_list_;

  typedef struct tag_sample_value{
      int       type;
      int       idx;
      int       idx_loc;
      T         dist;
  }SAMPLE_VALUE, *PSAMPLE_VALUE;

  typedef union tag_sample_data{
      float4         data_f;
      SAMPLE_VALUE   val;
  }SAMPLE_DATA, *PSAMPLE_DATA;

  const int                         max_nbor_size_;
  const int                         nnei_;
  const T                           rcut_;
  const Vector<int>                 sec_;

  template<typename FType, typename FVType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static inline int cal_distV(int const*      nlist,
                              FVType const*   npos,
                              int const*      type, 
                              int             nnumneighV, 
                              FVType const&   posi_x,
                              FVType const&   posi_y,
                              FVType const&   posi_z,
                              SAMPLE_DATA*    sel_data,
                              FVType          rcutV)
  {
    int    valid_cnt = 0;
    for (int j = 0; j < nnumneighV; ++j) 
    {
      int    j0 = j << 1;
      int    j1 = (j << 1) + 1;
      FVType x  = npos[3 * j];
      FVType y  = npos[3 * j + 1];
      FVType z  = npos[3 * j + 2];

      FVType a  = { x[0], y[1] };
      FVType b  = { x[1], z[0] };
      FVType c  = { y[0], z[1] };

      FVType distA = a - posi_x;
      FVType distB = b - posi_y;
      FVType distC = c - posi_z;

      distA = distA * distA;
      distB = distB * distB;
      distC = distC * distC;

      //FVType rr      = ipu::sqrt(distA + distB + distC);
      FVType      rr      = distA + distB + distC;
      int2        cmp_res = (int2)(rr <= rcutV);
      SAMPLE_DATA cur_res;
      cur_res.val.type    = type[j0];
      cur_res.val.idx     = nlist[j0];
      cur_res.val.idx_loc = j0;
      cur_res.val.dist    = rr[0];
      if (0 != cmp_res[0])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
      cur_res.val.type    = type[j1];
      cur_res.val.idx     = nlist[j1];
      cur_res.val.idx_loc = j1;
      cur_res.val.dist    = rr[1];
      if (0 != cmp_res[1])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
    }
    return valid_cnt;
  }

  template<typename FType, typename FVType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static inline int cal_distV(int const*        nlist,
                              FVType const*     npos,
                              int const*        type, 
                              int               nnumneighV, 
                              FVType const&     posi_x,
                              FVType const&     posi_y,
                              FVType const&     posi_z,
                              SAMPLE_DATA*      sel_data,
                              FVType            rcutV)
  {
    int    valid_cnt = 0;
    for (int j = 0; j < nnumneighV; ++j) 
    {
      int    j0 = j << 2;
      int    j1 = (j << 2) + 1;
      int    j2 = (j << 2) + 2;
      int    j3 = (j << 2) + 3;

      FVType x  = npos[3 * j];
      FVType y  = npos[3 * j + 1];
      FVType z  = npos[3 * j + 2];

      FVType a  = { x[0], x[3], y[2], z[1] };
      FVType b  = { x[1], y[0], y[3], z[2]};
      FVType c  = { x[2], y[1], z[0], z[3] };

      FVType distA = a - posi_x;
      FVType distB = b - posi_y;
      FVType distC = c - posi_z;

      distA = distA * distA;
      distB = distB * distB;
      distC = distC * distC;

      //FVType rr      = ipu::sqrt(distA + distB + distC);
      FVType rr           = distA + distB + distC;
      short4 cmp_res      = (short4)(rr <= rcutV);
      SAMPLE_DATA cur_res;
      cur_res.val.type    = type[j0];
      cur_res.val.idx     = nlist[j0];
      cur_res.val.idx_loc = j0;
      cur_res.val.dist    = rr[0];
      if (0 != cmp_res[0])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
      cur_res.val.type    = type[j1];
      cur_res.val.idx     = nlist[j1];
      cur_res.val.idx_loc = j1;
      cur_res.val.dist    = rr[1];
      if (0 != cmp_res[1])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
      cur_res.val.type    = type[j2];
      cur_res.val.idx     = nlist[j2];
      cur_res.val.idx_loc = j2;
      cur_res.val.dist    = rr[2];
      if (0 != cmp_res[2])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
      cur_res.val.type    = type[j3];
      cur_res.val.idx     = nlist[j3];
      cur_res.val.idx_loc = j3;
      cur_res.val.dist    = rr[3];
      if (0 != cmp_res[3])
      {
        sel_data[valid_cnt].data_f   = cur_res.data_f;
        valid_cnt ++;
      } 
    }
    return valid_cnt;
  }

  template <typename DT>
  static inline int qsort(SAMPLE_DATA* sample_data, int low, int high, int st_pos, int* stack_buf)
  {
    SAMPLE_DATA cur_data = sample_data[low];
    int         i        = low;
    int         j        = high;
    while(i < j)
    {
      while(i < j && (!((sample_data[j].val.type < cur_data.val.type) || 
                        (sample_data[j].val.type == cur_data.val.type && 
                        (sample_data[j].val.dist < cur_data.val.dist || 
                        (sample_data[j].val.dist == cur_data.val.dist && sample_data[j].val.idx < cur_data.val.idx))))))  j --;
      sample_data[i].data_f = sample_data[j].data_f;
      while(i < j &&  (!((cur_data.val.type < sample_data[i].val.type) || 
                        (cur_data.val.type == sample_data[i].val.type && 
                        (cur_data.val.dist < sample_data[i].val.dist || 
                        (cur_data.val.dist == sample_data[i].val.dist && cur_data.val.idx < sample_data[i].val.idx))))))  i ++;
      sample_data[j].data_f = sample_data[i].data_f;
    }
    sample_data[i].data_f = cur_data.data_f;

    if(low < i - 1)
    {
      stack_buf[2 * st_pos]     = low;
      stack_buf[2 * st_pos + 1] = i - 1;
      st_pos ++;
    }

    if(high > i + 1)
    {
      stack_buf[2 * st_pos]     = i + 1;
      stack_buf[2 * st_pos + 1] = high;
      st_pos ++;
    }

    return st_pos;
  };

  template <typename DT>
  static inline void isort(SAMPLE_DATA* sample_data, int size) 
  {
    for (int i = 1; i < size; ++i) 
    {
      SAMPLE_DATA cur_data;
      cur_data.data_f = sample_data[i].data_f;
      for (int j = i - 1; j >= 0; --j) 
      {
        SAMPLE_DATA cur_cursor_data;
        cur_cursor_data.data_f = sample_data[j].data_f;
        if (cur_data.val.type < cur_cursor_data.val.type || 
            (cur_data.val.type == cur_cursor_data.val.type && 
            (cur_data.val.dist < cur_cursor_data.val.dist || 
            (cur_data.val.dist == cur_cursor_data.val.dist && cur_data.val.idx < cur_cursor_data.val.idx))))
        {
          sample_data[j + 1].data_f = cur_cursor_data.data_f;
          if (0 == j)
          {
            sample_data[0].data_f = cur_data.data_f;
          }
        }
        else 
        {
          sample_data[j + 1].data_f = cur_data.data_f;
          break;
        }
      }
    }
  };

  template <typename DT>
  static inline void sort(SAMPLE_DATA* sample_data, 
                          int          size, 
                          int*         stack_buf)
  {
    int  cur_st_pos = 0;
    int  low        = 0;
    int  high       = size - 1;

    stack_buf[2 * cur_st_pos]     = low;
    stack_buf[2 * cur_st_pos + 1] = high;
    cur_st_pos ++;

    while(cur_st_pos > 0)
    {
      cur_st_pos--;
      low      = stack_buf[2 * cur_st_pos];
      high     = stack_buf[2 * cur_st_pos + 1];

      int  cur_len = high - low;
      if(cur_len >= 12)
        cur_st_pos = qsort<DT>(sample_data, low, high, cur_st_pos, stack_buf);
      else
        isort<DT>(sample_data + low, high - low + 1);
    }
  };

  static inline void cal_rij(int const*    nlist,
                             int const*    nnumneigh_size,
                             T const*      npos,
                             int const*    ntype,
                             T const*      pos_loc,
                             int           loop,
                             int           max_nbor_size,
                             int           nnei,
                             T             rcut,
                             int const*    sec,
                             int           sec_size,
                             int*          sel_idx_buf,
                             int*          stack_buf,
                             int*          fmt_idx_list,
                             T*            rij)
  {
    int const*       nlist_ptr        = nlist;
    T const*         npos_ptr         = npos;
    int const*       ntype_ptr        = ntype;
    int*             fmt_idx_ptr      = fmt_idx_list;
    SAMPLE_DATA*     sel_sample_ptr   = (SAMPLE_DATA*)sel_idx_buf;
    int*             sel_sec_ptr      = sel_idx_buf + 4 * max_nbor_size;
    int*             sel_fmt_loc_idx  = sel_idx_buf + 4 * max_nbor_size + nnei;
    T*               rij_ptr          = rij;
    T                rcut2            = rcut * rcut;
    typename FloatDef<T>::FVType         rcutV     = FloatDef<T>::setV(rcut2);
    for(int i = 0 ; i < loop ; i ++)
    {
      T    posi_x   = pos_loc[i * 3];
      T    posi_y   = pos_loc[i * 3 + 1];
      T    posi_z   = pos_loc[i * 3 + 2];
      typename FloatDef<T>::FVType posi_x_V = FloatDef<T>::setV(posi_x);
      typename FloatDef<T>::FVType posi_y_V = FloatDef<T>::setV(posi_y);
      typename FloatDef<T>::FVType posi_z_V = FloatDef<T>::setV(posi_z);
      typename FloatDef<T>::FVType const*  nposV_ptr = (typename FloatDef<T>::FVType const*)npos_ptr;
      int  nnumneigh  = nnumneigh_size[i];
      int  nnumneighV = nnumneigh >> FloatDef<T>::kSftBits;
      int  nnumneighA = nnumneighV << FloatDef<T>::kSftBits;
      int  valid_cnt  = cal_distV<T, typename FloatDef<T>::FVType>(
                                    nlist_ptr,
                                    nposV_ptr,
                                    ntype_ptr,
                                    nnumneighV, 
                                    posi_x_V, posi_y_V, posi_z_V,
                                    sel_sample_ptr, 
                                    rcutV);
      for(int j = nnumneighA ; j < nnumneigh ; ++ j)
      {
        T   a     = npos_ptr[3 * j]     - posi_x;
        T   b     = npos_ptr[3 * j + 1] - posi_y;
        T   c     = npos_ptr[3 * j + 2] - posi_z;
        //T   rr    = ipu::sqrt(a*a + b*b + c*c);
        T   rr    = a*a + b*b + c*c;
        SAMPLE_DATA cur_res;
        cur_res.val.type    = ntype_ptr[j];
        cur_res.val.idx     = nlist_ptr[j];
        cur_res.val.idx_loc = j;
        cur_res.val.dist    = rr;
        if (rr <= rcut2)
        {
          sel_sample_ptr[valid_cnt].data_f = cur_res.data_f;
          valid_cnt ++;
        }
      }
      sort<T>(sel_sample_ptr, valid_cnt, stack_buf);

      for (int j = 0 ; j < sec_size ; ++ j)
        sel_sec_ptr[j] = sec[j];
      for (int j = 0; j < valid_cnt; ++j) 
      {
        int nei_type = sel_sample_ptr[j].val.type;
        if (sel_sec_ptr[nei_type] < sec[nei_type+1])
        {
          int cur_idx = sel_sec_ptr[nei_type];
          fmt_idx_ptr[cur_idx]     = sel_sample_ptr[j].val.idx;
          sel_fmt_loc_idx[cur_idx] = sel_sample_ptr[j].val.idx_loc;
          cur_idx ++;
          sel_sec_ptr[nei_type] = cur_idx;
        }
      }
      for (int ii = 0; ii < sec_size - 1; ++ii) 
      {
        for (int jj = sec[ii]; jj < sec[ii + 1]; ++jj) 
        {
            if (fmt_idx_ptr[jj] < 0) 
              break;
            int j_idx = sel_fmt_loc_idx[jj];
            T   a     = npos_ptr[j_idx * 3];
            T   b     = npos_ptr[j_idx * 3 + 1];
            T   c     = npos_ptr[j_idx * 3 + 2];
            a         = a - posi_x;
            b         = b - posi_y;
            c         = c - posi_z;
            rij_ptr[jj * 3]     = a;
            rij_ptr[jj * 3 + 1] = b;
            rij_ptr[jj * 3 + 2] = c;
        }
      }

      nlist_ptr    += max_nbor_size;
      npos_ptr     += 3 * max_nbor_size;
      ntype_ptr    += max_nbor_size;
      fmt_idx_ptr  += nnei;
      rij_ptr      += 3 * nnei;
    }
  };

  bool compute() {
    int     loop          = fmt_idx_list_.size();
    int     desc_cnt      = nnei_ << 2;
    int     desc_cnt_V    = desc_cnt >> FloatDef<T>::kSftBits;
    loop = loop / nnei_; 
    int const*    nlist_ptr         = (int const*)(&(nlist_[0]));
    int const*    nnumneib_size_ptr = (int const*)(&(nnumneigh_size_[0]));
    T const*      npos_ptr          = (T const*)(&(npos_[0]));
    int const*    ntype_ptr         = (int const*)(&(ntype_[0]));
    T const*      pos_loc_ptr       = (T const*)(&(pos_loc_[0]));
    int*          fmt_idx_ptr       = (int*)(&fmt_idx_list_[0]);
    T*            rij_ptr           = (T*)(&rij_[0]);
    int const*    sec_ptr           = (int const*)(&(sec_[0]));
    int*          sel_idx_ptr       = (int*)(&(sel_idx_buf_[0]));
    int           stack[64]         = { 0 };
    cal_rij(nlist_ptr, 
            nnumneib_size_ptr,
            npos_ptr, 
            ntype_ptr, 
            pos_loc_ptr,
            loop, 
            max_nbor_size_, 
            nnei_, 
            rcut_, 
            sec_ptr, 
            sec_.size(), 
            sel_idx_ptr, 
            stack,
            fmt_idx_ptr, 
            rij_ptr);
    return true;
  };
};

template <typename T>
class ProdEnvMatVertex : public Vertex {
public:
  ProdEnvMatVertex();
  Input<Vector<int, ONE_PTR, 4>>    type_loc_;
  Input<Vector<T,   ONE_PTR, 8>>    rij_;
  Input<Vector<T,   ONE_PTR, 8>>    avg_;
  Input<Vector<T,   ONE_PTR, 8>>    std_;
  Input<Vector<int,   SPAN, 4>>     fmt_idx_list_;
  Output<Vector<T,  ONE_PTR, 8>>    env_mat_;
  Output<Vector<T,  ONE_PTR, 8>>    env_mat_deriv_;
  
  const int                         nnei_;
  const T                           rcut_smooth_;
  const T                           rcut_;
  const Vector<int>                 sec_;

  template<typename FType>
  static inline void spline5_switch(FType& vv, FType& dd, FType const& xx, FType const& rmin, FType const& rmax, FType max_min_inv)
  {
    if (xx < rmin) 
    {
      dd = 0.0f;
      vv = 1.0f;
    }
    else if (xx < rmax) 
    {
      //FType uu  = (xx - rmin) / (rmax - rmin) ;
      //FType du  = 1.0f / (rmax - rmin) ;
      FType uu  = (xx - rmin) * max_min_inv ;
      FType du  = 1.0f * max_min_inv ;
      FType val = (-6.0f * uu*uu + 15.0f * uu - 10.0f) ;
      vv        =      uu*uu*uu * val + 1.0f;
      dd        = (3.0f * uu*uu * val + uu*uu*uu * (-12.0f * uu + 15.0f)) * du;
    }
    else 
    {
      dd = 0.0f;
      vv = 0.0f;
    }
  }

  template<typename FType, typename FVType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void env_mat(int const*     fmt_idx_list,
                      FType const*   rij,
                      FVType const*  avg,
                      FVType const*  std,
                      int const*     sec,
                      int            sec_size,
                      int            nnei,
                      FType          rmin,
                      FType          rmax,
                      FType          max_min_inv,
                      FVType*        env_mat,
                      FVType*        env_mat_deriv)
  {
    for (int sec_iter = 0; sec_iter < sec_size - 1; ++sec_iter) 
    {
      for (int nei_iter = sec[sec_iter]; nei_iter < sec[sec_iter+1]; ++nei_iter) 
      {      
          if (fmt_idx_list[nei_iter] < 0) break;

          FType  rr_x     = rij[nei_iter * 3];
          FType  rr_y     = rij[nei_iter * 3 + 1];
          FType  rr_z     = rij[nei_iter * 3 + 2];

          FType  nr2   = rr_x * rr_x + rr_y * rr_y + rr_z * rr_z;
          FType  inr   = ipu::rsqrt(nr2);//1.0f/ipu::rsqrt(nr2);
          FType  nr    = nr2 * inr;
          FType  inr2  = inr * inr;
          FType  inr4  = inr2 * inr2;
          FType  inr3  = inr4 * nr;
          FType  sw, dsw;
          spline5_switch<FType>(sw, dsw, nr, rmin, rmax, max_min_inv);
          int idx_deriv = nei_iter * 2 * 3;	// 4 components time 3 directions
          int idx_value = nei_iter * 2;	    // 4 components
          // 4 value components

          FVType  avg0   = avg[idx_value];
          FVType  avg1   = avg[idx_value + 1];
          FVType  std0   = 1.0f / std[idx_value];
          FVType  std1   = 1.0f / std[idx_value + 1];
          FVType  std001 = { std0[0], std0[1] };
          FVType  std101 = { std1[0], std1[1] };

          FType   nr2_inv    = 1.0f / nr2;
          FVType  env_data_a = { 1.0f/ nr,        rr_x * nr2_inv };
          FVType  env_data_b = { rr_y * nr2_inv,  rr_z * nr2_inv };
          
          FVType  rr_x_y     = { rr_x,            rr_y };
          FVType  rr_y_z     = { rr_y,            rr_z };

          FVType  env_deriv_data_a_l = rr_x_y;
          FVType  env_deriv_data_b_l = { rr_z * inr3,               2.0f * rr_x * rr_x * inr4 - inr2 };
          FVType  env_deriv_data_c_l = rr_y_z;
          FVType  env_deriv_data_d_l = { 2.0f * rr_y * rr_x * inr4, 2.0f * rr_y * rr_y * inr4 - inr2 };
          FVType  env_deriv_data_e_l = { rr_y,                      rr_x };
          FVType  env_deriv_data_f_l = { 2.0f * rr_z * rr_y * inr4, 2.0f * rr_z * rr_z * inr4 - inr2 };
          env_deriv_data_a_l         = env_deriv_data_a_l * inr3;
          env_deriv_data_c_l         = 2.0f * rr_x * env_deriv_data_c_l * inr4;
          env_deriv_data_e_l         = 2.0f * env_deriv_data_e_l * rr_z * inr4;

          FType   dsw_x_inr          = dsw * inr;
          FVType  env_deriv_data_a_r = rr_x_y;
          FVType  env_deriv_data_b_r = { env_data_a[0] * rr_z, env_data_a[1] * rr_x };
          FVType  env_deriv_data_c_r = rr_y_z;
          FVType  env_deriv_data_d_r = rr_x_y;
          FVType  env_deriv_data_e_r = { env_data_b[0]* rr_z,  env_data_b[1] * rr_x };
          FVType  env_deriv_data_f_r = rr_y_z;
          env_deriv_data_a_r         = env_data_a[0] * env_deriv_data_a_r * dsw_x_inr;
          env_deriv_data_b_r         = env_deriv_data_b_r * dsw_x_inr;
          env_deriv_data_c_r         = env_data_a[1] * env_deriv_data_c_r * dsw_x_inr;
          env_deriv_data_d_r         = env_data_b[0] * env_deriv_data_d_r * dsw_x_inr;
          env_deriv_data_e_r         = env_deriv_data_e_r * dsw_x_inr;
          env_deriv_data_f_r         = env_data_b[1] * env_deriv_data_f_r * dsw_x_inr;

          env_deriv_data_a_l         = sw * env_deriv_data_a_l - env_deriv_data_a_r;
          env_deriv_data_b_l         = sw * env_deriv_data_b_l - env_deriv_data_b_r;
          env_deriv_data_c_l         = sw * env_deriv_data_c_l - env_deriv_data_c_r;
          env_deriv_data_d_l         = sw * env_deriv_data_d_l - env_deriv_data_d_r;
          env_deriv_data_e_l         = sw * env_deriv_data_e_l - env_deriv_data_e_r;
          env_deriv_data_f_l         = sw * env_deriv_data_f_l - env_deriv_data_f_r;

          env_mat_deriv[idx_deriv]     = env_deriv_data_a_l * std0[0];
          env_mat_deriv[idx_deriv + 1] = env_deriv_data_b_l * std001;
          env_mat_deriv[idx_deriv + 2] = env_deriv_data_c_l * std0[1];
          env_mat_deriv[idx_deriv + 3] = env_deriv_data_d_l * std1[0];
          env_mat_deriv[idx_deriv + 4] = env_deriv_data_e_l * std101;
          env_mat_deriv[idx_deriv + 5] = env_deriv_data_f_l * std1[1];

          env_data_a             = sw * env_data_a;
          env_data_b             = sw * env_data_b;

          env_mat[idx_value]     = (env_data_a - avg0) * std0;
          env_mat[idx_value + 1] = (env_data_b - avg1) * std1;
      }
    }
  }

  bool compute() {
    int     loop          = fmt_idx_list_.size();
    int     desc_cnt      = nnei_ << 2;
    int     desc_cnt_V    = desc_cnt >> FloatDef<T>::kSftBits;
    loop = loop / nnei_; 
    int const*                          fmt_idx_ptr       = (int const*)(&fmt_idx_list_[0]);
    T const*                            rij_ptr           = (T const*)(&rij_[0]);
    T                                   max_min_inv       = 1.0f / (rcut_ - rcut_smooth_) ;
    int const*                          type_loc_ptr      = (int const*)(&type_loc_[0]);
    typename FloatDef<T>::FVType*       env_mat_ptr       = (typename FloatDef<T>::FVType*)(&env_mat_[0]);
    typename FloatDef<T>::FVType*       env_mat_deriv_ptr = (typename FloatDef<T>::FVType*)(&env_mat_deriv_[0]);
    typename FloatDef<T>::FVType const* avg_ptr           = (typename FloatDef<T>::FVType const*)(&avg_[0]);
    typename FloatDef<T>::FVType const* std_ptr           = (typename FloatDef<T>::FVType const*)(&std_[0]);
    for(int i = 0 ; i < loop ; i ++)
    {
      int  norm_offset = type_loc_ptr[i] * desc_cnt_V;
      env_mat<T, typename FloatDef<T>::FVType>(fmt_idx_ptr,
                                               rij_ptr,
                                               avg_ptr + norm_offset,
                                               std_ptr + norm_offset,
                                               (int const*)(&sec_[0]),
                                               sec_.size(),
                                               nnei_,
                                               rcut_smooth_,
                                               rcut_,
                                               max_min_inv,
                                               env_mat_ptr,
                                               env_mat_deriv_ptr);
      fmt_idx_ptr       += nnei_;
      rij_ptr           += 3 * nnei_;
      env_mat_ptr       += desc_cnt_V;
      env_mat_deriv_ptr += 3 * desc_cnt_V;
    }
    return true;
  }
};

template class GenerateDistVertex<float>;
template class GenerateDistVertex<half>;
template class ProdEnvMatVertex<float>;
//template class ProdEnvMatVertex<half>;
