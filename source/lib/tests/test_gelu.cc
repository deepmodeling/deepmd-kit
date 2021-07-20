#include "gelu.h"
#include "device.h"
#include <iostream>
#include <gtest/gtest.h>
#include "utilities.h"

class TestGelu : public ::testing::Test
{
protected:
  // xx = tf.random.uniform([100], minval=-4, maxval=4, dtype = tf.float64)
  std::vector<double > xx = {
    -0.65412617, -0.74208893, -2.21731157,  0.42540039, -0.20889174, -1.37948692,
     3.36373004, -2.51647562, -2.985111  , -0.53251562,  0.36373729, -3.19052875,
     0.37908265,  0.81605825,  1.66281318,  2.71761869, -0.89313006,  0.11503315,
     2.2010268 ,  0.65498149,  1.51153638,  0.71501482, -1.27392131, -2.89503271,
     0.53546578,  1.4564317 , -2.36701143,  1.23009056,  0.20264839,  2.06037292,
     3.41302551, -2.3175205 , -0.27628221, -1.35701656,  2.13781656, -0.52921087,
    -1.56774526,  2.92475766, -3.17376756, -2.61726505, -0.89399621, -1.30025318,
    -3.98310127,  0.0378038 , -0.59195525, -0.71764632,  2.83774732, -1.83266476,
    -3.52590216,  2.7735313 , -1.52387184, -3.57984618,  3.44036277, -1.52546413,
     3.34095372,  1.12462909,  0.64319821, -3.94019443, -0.4976394 , -3.84744725,
     3.32683062, -1.95707363, -1.73538352, -3.32614596, -1.46374614,  3.32600174,
     1.56399235,  3.42035556,  3.029476  , -3.99135473,  2.22774417, -2.27991379,
    -0.12769364,  3.27522847, -1.18421457, -2.65598248,  0.37112235,  1.27091438,
     1.82646907,  2.06702457,  2.87834558,  0.63158531, -1.76016732, -0.85704887,
     1.07093632, -2.55155726,  0.60068505, -0.36984068, -1.75685256,  1.2808404,
     3.07005843,  1.11521146,  2.3648244 , -2.79509595,  2.4611316 ,  2.95155864,
     3.45913518,  2.71155262,  0.49731474,  0.89416884
  };
  std::vector<double > expected_gelu = {
    -1.67837557e-01, -1.70017454e-01, -2.92128115e-02,  2.82765887e-01,
    -8.71641062e-02, -1.15934278e-01,  3.36269232e+00, -1.44692661e-02,
    -3.81342874e-03, -1.58276988e-01,  2.33504238e-01, -1.93195840e-03,
     2.45520665e-01,  6.46854620e-01,  1.58255340e+00,  2.70915699e+00,
    -1.66142311e-01,  6.27839507e-02,  2.17077193e+00,  4.87104731e-01,
     1.41257916e+00,  5.45282609e-01, -1.29333636e-01, -5.04228492e-03,
     3.76858089e-01,  1.35041498e+00, -2.08435518e-02,  1.09538283e+00,
     1.17595324e-01,  2.01997211e+00,  3.41216324e+00, -2.33762781e-02,
    -1.08073931e-01, -1.18820554e-01,  2.10325928e+00, -1.57900404e-01,
    -9.18635121e-02,  2.92015365e+00, -2.04685946e-03, -1.11316220e-02,
    -1.66096393e-01, -1.26039117e-01, -7.61243780e-05,  1.94719045e-02,
    -1.63967673e-01, -1.69774465e-01,  2.83175981e+00, -6.13003406e-02,
    -5.56239423e-04,  2.76631042e+00, -9.73891862e-02, -4.47898619e-04,
     3.43958593e+00, -9.71871941e-02,  3.33982471e+00,  9.77813916e-01,
     4.75894192e-01, -9.31449548e-05, -1.53971187e-01, -1.42502838e-04,
     3.32564148e+00, -4.91974866e-02, -7.18453399e-02, -1.19212505e-03,
    -1.05075731e-01,  3.32480898e+00,  1.47165971e+00,  3.41951698e+00,
     3.02616656e+00, -7.31989124e-05,  2.19918490e+00, -2.54527487e-02,
    -5.73595208e-02,  3.27379481e+00, -1.40141518e-01, -1.00297107e-02,
     2.39266857e-01,  1.14120736e+00,  1.76452433e+00,  2.02715079e+00,
     2.87304205e+00,  4.64915551e-01, -6.90746681e-02, -1.67834746e-01,
     9.18580320e-01, -1.32266764e-02,  4.36049337e-01, -1.31576637e-01,
    -6.94420205e-02,  1.15236826e+00,  3.06715854e+00,  9.67388918e-01,
     2.34387360e+00, -6.78489313e-03,  2.44451366e+00,  2.94732148e+00,
     3.45841254e+00,  2.70294624e+00,  3.43387119e-01,  7.28081631e-01
  };  
  std::vector<double > expected_gelu_grad = {
     4.60449412e-02,  4.50599718e-03, -6.31675690e-02,  8.19672783e-01,
     3.35740612e-01, -1.28672776e-01,  1.00385962e+00, -3.67087198e-02,
    -1.20649335e-02,  1.12970546e-01,  7.77739313e-01, -6.68287407e-03,
     7.88372245e-01,  1.02580252e+00,  1.11883773e+00,  1.02368320e+00,
    -5.28753300e-02,  5.91377555e-01,  1.06481455e+00,  9.54387332e-01,
     1.12733634e+00,  9.83340637e-01, -1.24278352e-01, -1.53213139e-02,
     8.88777668e-01,  1.12870635e+00, -4.89412880e-02,  1.12076578e+00,
     6.59486509e-01,  1.07959136e+00,  1.00327260e+00, -5.34451698e-02,
     2.85098027e-01, -1.28185394e-01,  1.07135120e+00,  1.14935972e-01,
    -1.24907600e-01,  1.01417860e+00, -7.02992824e-03, -2.96883138e-02,
    -5.31536587e-02, -1.25891871e-01, -3.60887857e-04,  5.30148635e-01,
     7.89229070e-02,  1.54532953e-02,  1.01772418e+00, -1.03699345e-01,
    -2.20978811e-03,  1.02074891e+00, -1.26887416e-01, -1.81810307e-03,
     1.00298141e+00, -1.26825892e-01,  1.00415984e+00,  1.10771369e+00,
     9.48383787e-01, -4.34543288e-04,  1.34084905e-01, -6.41896044e-04,
     1.00435580e+00, -9.07188185e-02, -1.12900642e-01, -4.36549981e-03,
    -1.28587248e-01,  1.00436754e+00,  1.12509925e+00,  1.00319222e+00,
     1.01067764e+00, -3.48088477e-04,  1.06212145e+00, -5.70046708e-02,
     3.98669653e-01,  1.00513974e+00, -1.15918449e-01, -2.72566889e-02,
     7.82872230e-01,  1.12407088e+00,  1.10431752e+00,  1.07887693e+00,
     1.01599352e+00,  9.42365429e-01, -1.10671686e-01, -4.07709087e-02,
     1.09835643e+00, -3.41514078e-02,  9.25865272e-01,  2.18016504e-01,
    -1.10974960e-01,  1.12473750e+00,  1.00952394e+00,  1.10621038e+00,
     1.04913579e+00, -1.96928674e-02,  1.04098891e+00,  1.01320664e+00,
     1.00279462e+00,  1.02401893e+00,  8.65714727e-01,  1.05320906e+00
  };  
  std::vector<double > expected_gelu_grad_grad = {
     0.50571564,  0.43843355, -0.10061714,  0.66240156,  0.76347293,  0.01663728,
    -0.01276427, -0.07462592, -0.03281601,  0.59359609,  0.69699731, -0.02027303,
     0.68877611,  0.38103445, -0.07490714, -0.05506099,  0.32166971,  0.78732762,
    -0.10164498,  0.5050721 , -0.03437986,  0.4593321 ,  0.06819688, -0.0396137,
     0.59156916, -0.01491246, -0.08883747,  0.09234241,  0.7654675 , -0.10747085,
    -0.01108326, -0.09311994,  0.7384317 ,  0.02681523, -0.10499993,  0.59585922,
    -0.05160053, -0.03728553, -0.02114303, -0.06469217,  0.32100942,  0.05445215,
    -0.00157624,  0.79673926,  0.55165186,  0.45730633, -0.04432554, -0.10006544,
    -0.00789302, -0.04993342, -0.03838479, -0.00665763, -0.01022962, -0.03889244,
    -0.01360533,  0.1565692 ,  0.51391336, -0.00186407,  0.61706551, -0.00264692,
    -0.01414804, -0.10721734, -0.08807903, -0.01417477, -0.01764587, -0.0141804,
    -0.05053224, -0.01084901, -0.02975642, -0.00152558, -0.09992717, -0.09614436,
     0.7848912 , -0.01627357,  0.11925413, -0.06092512,  0.69307417,  0.06980313,
    -0.09948152, -0.10733955, -0.04095624,  0.52257218, -0.09172304,  0.34933309,
     0.19229249, -0.07116424,  0.54531393,  0.6937595 , -0.09125988,  0.06452926,
    -0.02712756,  0.16269737, -0.08903289, -0.04801427, -0.08003338, -0.03525758,
    -0.00967444, -0.05562922,  0.61727952,  0.32087784
  }; 

  const int nloc = xx.size();

  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(TestGelu, gelu_cpu)
{
  std::vector<double> gelu(nloc);
  deepmd::gelu_cpu<double> (&gelu[0], &xx[0], nloc);
  EXPECT_EQ(gelu.size(), nloc);
  EXPECT_EQ(gelu.size(), expected_gelu.size());
  for (int jj = 0; jj < gelu.size(); ++jj){
    EXPECT_LT(fabs(gelu[jj] - expected_gelu[jj]) , 1e-5);
  }  
}

TEST_F(TestGelu, gelu_grad_cpu)
{
  std::vector<double> dy(100, 1.0);
  std::vector<double> gelu_grad(nloc);
  deepmd::gelu_grad_cpu<double> (&gelu_grad[0], &xx[0], &dy[0], nloc);
  EXPECT_EQ(gelu_grad.size(), nloc);
  EXPECT_EQ(gelu_grad.size(), expected_gelu_grad.size());
  for (int jj = 0; jj < gelu_grad.size(); ++jj){
    EXPECT_LT(fabs(gelu_grad[jj] - expected_gelu_grad[jj]) , 1e-5);
  }  
}

TEST_F(TestGelu, gelu_grad_grad_cpu)
{
  std::vector<double> dy(100, 1.0);
  std::vector<double> dy_2(100, 1.0);
  std::vector<double> gelu_grad_grad(nloc);
  deepmd::gelu_grad_grad_cpu<double> (&gelu_grad_grad[0], &xx[0], &dy[0], &dy_2[0], nloc);
  EXPECT_EQ(gelu_grad_grad.size(), nloc);
  EXPECT_EQ(gelu_grad_grad.size(), expected_gelu_grad_grad.size());
  for (int jj = 0; jj < gelu_grad_grad.size(); ++jj){
    EXPECT_LT(fabs(gelu_grad_grad[jj] - expected_gelu_grad_grad[jj]) , 1e-5);
  }  
}

#if GOOGLE_CUDA
TEST_F(TestGelu, gelu_gpu_cuda)
{
  std::vector<double> gelu(nloc, 0.0);
  
  double * gelu_dev = NULL, * xx_dev = NULL;
  deepmd::malloc_device_memory_sync(gelu_dev, gelu);
  deepmd::malloc_device_memory_sync(xx_dev, xx);
  deepmd::gelu_gpu_cuda<double> (gelu_dev, xx_dev, nloc);
  deepmd::memcpy_device_to_host(gelu_dev, gelu);
  deepmd::delete_device_memory(gelu_dev);
  deepmd::delete_device_memory(xx_dev);

  EXPECT_EQ(gelu.size(), nloc);
  EXPECT_EQ(gelu.size(), expected_gelu.size());
  for (int jj = 0; jj < gelu.size(); ++jj){
    EXPECT_LT(fabs(gelu[jj] - expected_gelu[jj]) , 1e-5);
  }  
}

TEST_F(TestGelu, gelu_grad_gpu_cuda)
{
  std::vector<double> dy(100, 1.0);
  std::vector<double> gelu_grad(nloc, 0.0);

  double * gelu_grad_dev = NULL, * xx_dev = NULL, * dy_dev = NULL;
  deepmd::malloc_device_memory_sync(gelu_grad_dev, gelu_grad);
  deepmd::malloc_device_memory_sync(xx_dev, xx);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::gelu_grad_gpu_cuda<double> (gelu_grad_dev, xx_dev, dy_dev, nloc);
  deepmd::memcpy_device_to_host(gelu_grad_dev, gelu_grad);
  deepmd::delete_device_memory(gelu_grad_dev);
  deepmd::delete_device_memory(xx_dev);
  deepmd::delete_device_memory(dy_dev);

  EXPECT_EQ(gelu_grad.size(), nloc);
  EXPECT_EQ(gelu_grad.size(), expected_gelu_grad.size());
  for (int jj = 0; jj < gelu_grad.size(); ++jj){
    EXPECT_LT(fabs(gelu_grad[jj] - expected_gelu_grad[jj]) , 1e-5);
  }  
}

TEST_F(TestGelu, gelu_grad_grad_gpu_cuda)
{
  std::vector<double> dy(100, 1.0);
  std::vector<double> dy_2(100, 1.0);
  std::vector<double> gelu_grad_grad(nloc, 0.0);

  double * gelu_grad_grad_dev = NULL, * xx_dev = NULL, * dy_dev = NULL, * dy_2_dev = NULL;
  deepmd::malloc_device_memory_sync(gelu_grad_grad_dev, gelu_grad_grad);
  deepmd::malloc_device_memory_sync(xx_dev, xx);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::malloc_device_memory_sync(dy_2_dev, dy_2);
  deepmd::gelu_grad_grad_gpu_cuda<double> (gelu_grad_grad_dev, xx_dev, dy_dev, dy_2_dev, nloc);
  deepmd::memcpy_device_to_host(gelu_grad_grad_dev, gelu_grad_grad);
  deepmd::delete_device_memory(gelu_grad_grad_dev);
  deepmd::delete_device_memory(xx_dev);
  deepmd::delete_device_memory(dy_dev);
  deepmd::delete_device_memory(dy_2_dev);

  EXPECT_EQ(gelu_grad_grad.size(), nloc);
  EXPECT_EQ(gelu_grad_grad.size(), expected_gelu_grad_grad.size());
  for (int jj = 0; jj < gelu_grad_grad.size(); ++jj){
    EXPECT_LT(fabs(gelu_grad_grad[jj] - expected_gelu_grad_grad[jj]) , 1e-5);
  }  
}
#endif // GOOGLE_CUDA
