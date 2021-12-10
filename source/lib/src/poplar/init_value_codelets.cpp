#include <poplar/Vertex.hpp>
#include <ipudef.h>

using namespace poplar;

static constexpr auto SPAN    = poplar::VectorLayout::SPAN;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template<typename T>
struct DataDef{
};

template<>
struct DataDef<float>{
  static inline const int         kSftBits  = 1;
  typedef   float2                DVType;
  static inline DVType            setV(float const& a)
  {
    DVType x = { a, a };
    return x;
  };
};

template<>
struct DataDef<half>{
  static inline const int         kSftBits  = 2;
  typedef   half4                 DVType;
  static inline DVType            setV(half const& a)
  {
    DVType x = { a, a, a, a };
    return x;
  };
};

template<>
struct DataDef<int>{
  static inline const int         kSftBits  = 1;
  typedef   int2                  DVType;
  static inline DVType            setV(int const& a)
  {
    DVType x = { a, a, };
    return x;
  };
};

void InitMemoryFp32(float* dst, int data_cnt, float val);
void InitMemoryFp16(half*  dst, int data_cnt, half  val);
void InitMemoryInt32(int*  dst, int data_cnt, int   val);

template <class T> class InitValueVertex : public Vertex {
public:
  InitValueVertex();
  Output<Vector<T, SPAN, 8>>  data_;

  const T                     value_;

  template<typename FType, typename std::enable_if<std::is_same<FType, float>::value, void>::type* = nullptr>
  static void init_value(FType* dst, int data_cnt, FType val)
  {
    InitMemoryFp32(dst, data_cnt, val);
  };

  template<typename FType, typename std::enable_if<std::is_same<FType, half>::value, void>::type* = nullptr>
  static void init_value(FType* dst, int data_cnt, FType val)
  {
    InitMemoryFp16(dst, data_cnt, val);
    if(0 != (data_cnt&1))
      dst[(data_cnt >> 1) << 1] = val;
  };

  template<typename FType, typename std::enable_if<std::is_same<FType, int>::value, void>::type* = nullptr>
  static void init_value(FType* dst, int data_cnt, FType val)
  {
    InitMemoryInt32(dst, data_cnt, val);
  };

  bool compute() {
    int   data_size   = data_.size();
  #if 0
    int   data_size_q = data_size   >> DataDef<T>::kSftBits;
    int   data_size_v = data_size_q << DataDef<T>::kSftBits;
    int   i           = 0;
    typename DataDef<T>::DVType*  data_v_ptr = (typename DataDef<T>::DVType*)(&(data_[0]));
    T*                            data_ptr   = (T*)(&(data_[0]));
    typename DataDef<T>::DVType   value      = DataDef<T>::setV(value_);
    for(i = 0 ; i < data_size_q ; i ++)
      data_v_ptr[i] = value;
    for(i = data_size_v ; i < data_size ; i ++)
      data_ptr[i] = value_;
#else
    T*  data_ptr   = (T*)(&(data_[0]));
    init_value<T>(data_ptr, data_size, value_);
#endif
    return true;
  }
};

template class InitValueVertex<float>;
template class InitValueVertex<half>;
template class InitValueVertex<int>;