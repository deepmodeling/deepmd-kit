// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "device.h"
#include "tabulate.h"

namespace {

constexpr double kLower = -1.0;
constexpr double kUpper = 1.0;
constexpr double kMax = 2.0;
constexpr double kMin = -kMax;
constexpr double kStride0 = 1.0;
constexpr double kStride1 = 1.0;
constexpr int kLastLayerSize = 1;
constexpr double kFiniteDiffStep = 1e-6;
constexpr double kTolerance = 1e-10;

const std::vector<double> kTableInfo = {kLower,   kUpper,   kMax,
                                        kStride0, kStride1, -1.0};
const std::vector<double> kOffGridTableInfo = {kLower,   kUpper,   2.5,
                                               kStride0, kStride1, -1.0};

const std::vector<double> kTable = {
    10.0, 2.0,  3.0,  0.25, -0.5,  0.1,    //
    20.0, -1.0, 0.5,  -0.2, 0.25,  -0.05,  //
    30.0, 4.0,  -2.0, 1.0,  0.1,   0.2,    //
    40.0, -3.0, 1.0,  0.5,  -0.25, 0.05,
};

struct LocatedXx {
  int table_idx;
  double offset;
  double extrapolate_delta;
};

double poly5(const double* coeff, const double xx) {
  return coeff[0] +
         (coeff[1] +
          (coeff[2] + (coeff[3] + (coeff[4] + coeff[5] * xx) * xx) * xx) * xx) *
             xx;
}

double poly5_grad(const double* coeff, const double xx) {
  return coeff[1] +
         (2.0 * coeff[2] +
          (3.0 * coeff[3] + (4.0 * coeff[4] + 5.0 * coeff[5] * xx) * xx) * xx) *
             xx;
}

LocatedXx locate_se_a_or_r(const double xx,
                           const std::vector<double>& table_info) {
  const double lower = table_info[0];
  const double upper = table_info[1];
  const double max = table_info[2];
  const double stride0 = table_info[3];
  const double stride1 = table_info[4];
  if (xx < lower) {
    return {0, 0.0, xx - lower};
  }
  if (xx < upper) {
    const int table_idx = static_cast<int>((xx - lower) / stride0);
    return {table_idx, xx - (table_idx * stride0 + lower), 0.0};
  }
  if (xx < max) {
    const int first_stride = static_cast<int>((upper - lower) / stride0);
    const int table_idx =
        first_stride + static_cast<int>((xx - upper) / stride1);
    return {table_idx, xx - ((table_idx - first_stride) * stride1 + upper),
            0.0};
  }

  const int first_stride = static_cast<int>((upper - lower) / stride0);
  const double boundary_xx = std::nextafter(max, lower);
  const int table_idx =
      first_stride + static_cast<int>((boundary_xx - upper) / stride1);
  return {table_idx, max - ((table_idx - first_stride) * stride1 + upper),
          xx - max};
}

LocatedXx locate_se_a_or_r(const double xx) {
  return locate_se_a_or_r(xx, kTableInfo);
}

LocatedXx locate_se_t(const double xx, const std::vector<double>& table_info) {
  const double lower = table_info[0];
  const double upper = table_info[1];
  const double max = table_info[2];
  const double min = -max;
  const double stride0 = table_info[3];
  const double stride1 = table_info[4];
  if (xx < min) {
    return {0, 0.0, xx - min};
  }
  if (xx < lower) {
    const int table_idx = static_cast<int>((xx - min) / stride1);
    return {table_idx, xx - (table_idx * stride1 + min), 0.0};
  }
  if (xx < upper) {
    const int first_stride = static_cast<int>((lower - min) / stride1);
    const int table_idx =
        first_stride + static_cast<int>((xx - lower) / stride0);
    return {table_idx, xx - ((table_idx - first_stride) * stride0 + lower),
            0.0};
  }
  if (xx < max) {
    const int first_stride = static_cast<int>((lower - min) / stride1) +
                             static_cast<int>((upper - lower) / stride0);
    const int table_idx =
        first_stride + static_cast<int>((xx - upper) / stride1);
    return {table_idx, xx - ((table_idx - first_stride) * stride1 + upper),
            0.0};
  }

  const int first_stride = static_cast<int>((lower - min) / stride1) +
                           static_cast<int>((upper - lower) / stride0);
  const double boundary_xx = std::nextafter(max, min);
  const int table_idx =
      first_stride + static_cast<int>((boundary_xx - upper) / stride1);
  return {table_idx, max - ((table_idx - first_stride) * stride1 + upper),
          xx - max};
}

LocatedXx locate_se_t(const double xx) { return locate_se_t(xx, kTableInfo); }

double expected_table_value(const LocatedXx& located) {
  const double* coeff = &kTable[located.table_idx * 6];
  return poly5(coeff, located.offset) +
         poly5_grad(coeff, located.offset) * located.extrapolate_delta;
}

double expected_table_grad(const LocatedXx& located) {
  return poly5_grad(&kTable[located.table_idx * 6], located.offset);
}

double se_a_value(const double xx) {
  std::vector<double> out(4 * kLastLayerSize);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {1.0, 0.0, 0.0, 0.0};
  deepmd::tabulate_fusion_se_a_cpu<double>(
      out.data(), kTable.data(), kTableInfo.data(), em_x.data(), em.data(),
      nullptr, 1, 1, kLastLayerSize);
  return out[0];
}

double se_a_grad(const double xx) {
  std::vector<double> dy_dem_x(1);
  std::vector<double> dy_dem(4);
  std::vector<double> dy_dtwo(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> dy = {1.0, 0.0, 0.0, 0.0};
  deepmd::tabulate_fusion_se_a_grad_cpu<double>(
      dy_dem_x.data(), dy_dem.data(), dy_dtwo.data(), kTable.data(),
      kTableInfo.data(), em_x.data(), em.data(), nullptr, dy.data(), 1, 1,
      kLastLayerSize);
  return dy_dem_x[0];
}

double se_r_value(const double xx, const std::vector<double>& table_info) {
  std::vector<double> out(1);
  const std::vector<double> em = {xx};
  deepmd::tabulate_fusion_se_r_cpu(out.data(), kTable.data(), table_info.data(),
                                   em.data(), 1, 1, kLastLayerSize);
  return out[0];
}

double se_r_value(const double xx) { return se_r_value(xx, kTableInfo); }

double se_r_grad(const double xx, const std::vector<double>& table_info) {
  std::vector<double> dy_dem(1);
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  deepmd::tabulate_fusion_se_r_grad_cpu(dy_dem.data(), kTable.data(),
                                        table_info.data(), em.data(), dy.data(),
                                        1, 1, kLastLayerSize);
  return dy_dem[0];
}

double se_r_grad(const double xx) { return se_r_grad(xx, kTableInfo); }

double se_t_value(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  deepmd::tabulate_fusion_se_t_cpu(out.data(), kTable.data(), kTableInfo.data(),
                                   em_x.data(), em.data(), 1, 1, 1,
                                   kLastLayerSize);
  return out[0];
}

void se_t_grad(const double xx, double& dy_dem_x, double& dy_dem) {
  std::vector<double> dy_dem_x_vec(1);
  std::vector<double> dy_dem_vec(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  deepmd::tabulate_fusion_se_t_grad_cpu(
      dy_dem_x_vec.data(), dy_dem_vec.data(), kTable.data(), kTableInfo.data(),
      em_x.data(), em.data(), dy.data(), 1, 1, 1, kLastLayerSize);
  dy_dem_x = dy_dem_x_vec[0];
  dy_dem = dy_dem_vec[0];
}

double se_t_tebd_value(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  deepmd::tabulate_fusion_se_t_tebd_cpu(out.data(), kTable.data(),
                                        kTableInfo.data(), em_x.data(),
                                        em.data(), 1, 1, 1, kLastLayerSize);
  return out[0];
}

double se_t_tebd_grad(const double xx) {
  std::vector<double> dy_dem_x(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  deepmd::tabulate_fusion_se_t_tebd_grad_cpu(
      dy_dem_x.data(), kTable.data(), kTableInfo.data(), em_x.data(), em.data(),
      dy.data(), 1, 1, 1, kLastLayerSize);
  return dy_dem_x[0];
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
double se_a_value_gpu(const double xx) {
  std::vector<double> out(4 * kLastLayerSize);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {1.0, 0.0, 0.0, 0.0};
  double *out_dev = nullptr, *table_dev = nullptr, *em_x_dev = nullptr,
         *em_dev = nullptr;
  deepmd::malloc_device_memory_sync(out_dev, out);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::tabulate_fusion_se_a_gpu<double>(out_dev, table_dev,
                                           kTableInfo.data(), em_x_dev, em_dev,
                                           nullptr, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(out_dev, out);
  deepmd::delete_device_memory(out_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  return out[0];
}

double se_a_grad_gpu(const double xx) {
  std::vector<double> dy_dem_x(1);
  std::vector<double> dy_dem(4);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> dy = {1.0, 0.0, 0.0, 0.0};
  double *dy_dem_x_dev = nullptr, *dy_dem_dev = nullptr, *table_dev = nullptr,
         *em_x_dev = nullptr, *em_dev = nullptr, *dy_dev = nullptr;
  deepmd::malloc_device_memory_sync(dy_dem_x_dev, dy_dem_x);
  deepmd::malloc_device_memory_sync(dy_dem_dev, dy_dem);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::tabulate_fusion_se_a_grad_gpu<double>(
      dy_dem_x_dev, dy_dem_dev, nullptr, table_dev, kTableInfo.data(), em_x_dev,
      em_dev, nullptr, dy_dev, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(dy_dem_x_dev, dy_dem_x);
  deepmd::delete_device_memory(dy_dem_x_dev);
  deepmd::delete_device_memory(dy_dem_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(dy_dev);
  return dy_dem_x[0];
}

double se_r_value_gpu(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em = {xx};
  double *out_dev = nullptr, *table_dev = nullptr, *em_dev = nullptr;
  deepmd::malloc_device_memory_sync(out_dev, out);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::tabulate_fusion_se_r_gpu<double>(
      out_dev, table_dev, kTableInfo.data(), em_dev, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(out_dev, out);
  deepmd::delete_device_memory(out_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_dev);
  return out[0];
}

double se_r_grad_gpu(const double xx) {
  std::vector<double> dy_dem(1);
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  double *dy_dem_dev = nullptr, *table_dev = nullptr, *em_dev = nullptr,
         *dy_dev = nullptr;
  deepmd::malloc_device_memory_sync(dy_dem_dev, dy_dem);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::tabulate_fusion_se_r_grad_gpu<double>(dy_dem_dev, table_dev,
                                                kTableInfo.data(), em_dev,
                                                dy_dev, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(dy_dem_dev, dy_dem);
  deepmd::delete_device_memory(dy_dem_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(dy_dev);
  return dy_dem[0];
}

double se_t_value_gpu(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  double *out_dev = nullptr, *table_dev = nullptr, *em_x_dev = nullptr,
         *em_dev = nullptr;
  deepmd::malloc_device_memory_sync(out_dev, out);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::tabulate_fusion_se_t_gpu<double>(out_dev, table_dev,
                                           kTableInfo.data(), em_x_dev, em_dev,
                                           1, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(out_dev, out);
  deepmd::delete_device_memory(out_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  return out[0];
}

void se_t_grad_gpu(const double xx, double& dy_dem_x, double& dy_dem) {
  std::vector<double> dy_dem_x_vec(1);
  std::vector<double> dy_dem_vec(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  double *dy_dem_x_dev = nullptr, *dy_dem_dev = nullptr, *table_dev = nullptr,
         *em_x_dev = nullptr, *em_dev = nullptr, *dy_dev = nullptr;
  deepmd::malloc_device_memory_sync(dy_dem_x_dev, dy_dem_x_vec);
  deepmd::malloc_device_memory_sync(dy_dem_dev, dy_dem_vec);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::tabulate_fusion_se_t_grad_gpu<double>(
      dy_dem_x_dev, dy_dem_dev, table_dev, kTableInfo.data(), em_x_dev, em_dev,
      dy_dev, 1, 1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(dy_dem_x_dev, dy_dem_x_vec);
  deepmd::memcpy_device_to_host(dy_dem_dev, dy_dem_vec);
  deepmd::delete_device_memory(dy_dem_x_dev);
  deepmd::delete_device_memory(dy_dem_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(dy_dev);
  dy_dem_x = dy_dem_x_vec[0];
  dy_dem = dy_dem_vec[0];
}

double se_t_tebd_value_gpu(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  double *out_dev = nullptr, *table_dev = nullptr, *em_x_dev = nullptr,
         *em_dev = nullptr;
  deepmd::malloc_device_memory_sync(out_dev, out);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::tabulate_fusion_se_t_tebd_gpu<double>(
      out_dev, table_dev, kTableInfo.data(), em_x_dev, em_dev, 1, 1, 1,
      kLastLayerSize);
  deepmd::memcpy_device_to_host(out_dev, out);
  deepmd::delete_device_memory(out_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  return out[0];
}

double se_t_tebd_grad_gpu(const double xx) {
  std::vector<double> dy_dem_x(1);
  const std::vector<double> em_x = {xx};
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  double *dy_dem_x_dev = nullptr, *table_dev = nullptr, *em_x_dev = nullptr,
         *em_dev = nullptr, *dy_dev = nullptr;
  deepmd::malloc_device_memory_sync(dy_dem_x_dev, dy_dem_x);
  deepmd::malloc_device_memory_sync(table_dev, kTable);
  deepmd::malloc_device_memory_sync(em_x_dev, em_x);
  deepmd::malloc_device_memory_sync(em_dev, em);
  deepmd::malloc_device_memory_sync(dy_dev, dy);
  deepmd::tabulate_fusion_se_t_tebd_grad_gpu<double>(
      dy_dem_x_dev, table_dev, kTableInfo.data(), em_x_dev, em_dev, dy_dev, 1,
      1, 1, kLastLayerSize);
  deepmd::memcpy_device_to_host(dy_dem_x_dev, dy_dem_x);
  deepmd::delete_device_memory(dy_dem_x_dev);
  deepmd::delete_device_memory(table_dev);
  deepmd::delete_device_memory(em_x_dev);
  deepmd::delete_device_memory(em_dev);
  deepmd::delete_device_memory(dy_dev);
  return dy_dem_x[0];
}
#endif

double central_diff(double (*fn)(double), const double xx) {
  return (fn(xx + kFiniteDiffStep) - fn(xx - kFiniteDiffStep)) /
         (2.0 * kFiniteDiffStep);
}

double grad_central_diff(double (*fn)(double), const double xx) {
  return (fn(xx + kFiniteDiffStep) - fn(xx - kFiniteDiffStep)) /
         (2.0 * kFiniteDiffStep);
}

void expect_linear_tail(double (*value_fn)(double),
                        double (*grad_fn)(double),
                        const double xx,
                        const LocatedXx& located) {
  EXPECT_NEAR(value_fn(xx), expected_table_value(located), kTolerance);
  EXPECT_NEAR(grad_fn(xx), expected_table_grad(located), kTolerance);
  EXPECT_NEAR(central_diff(value_fn, xx), grad_fn(xx), 1e-8);
  EXPECT_NEAR(grad_central_diff(grad_fn, xx), 0.0, 1e-10);
}

void expect_boundary(double (*value_fn)(double),
                     double (*grad_fn)(double),
                     const double xx,
                     const LocatedXx& located) {
  EXPECT_NEAR(value_fn(xx), expected_table_value(located), kTolerance);
  EXPECT_NEAR(grad_fn(xx), expected_table_grad(located), kTolerance);
}

void expect_se_t_lookup_tail(double (*value_fn)(double),
                             void (*grad_fn)(double, double&, double&),
                             const double xx) {
  const LocatedXx located = locate_se_t(xx);
  const double expected_value = expected_table_value(located);
  const double expected_grad = expected_table_grad(located);
  double dy_dem_x = 0.0;
  double dy_dem = 0.0;
  grad_fn(xx, dy_dem_x, dy_dem);
  EXPECT_NEAR(value_fn(xx), xx * expected_value, kTolerance);
  EXPECT_NEAR(dy_dem_x, xx * expected_grad, kTolerance);
  EXPECT_NEAR(dy_dem, expected_value, kTolerance);
  EXPECT_NEAR(central_diff(value_fn, xx), dy_dem_x + dy_dem, 1e-5);
}

}  // namespace

TEST(TabulateExtrapolate, SeAUsesC1LinearTails) {
  expect_linear_tail(se_a_value, se_a_grad, kLower - 0.25,
                     locate_se_a_or_r(kLower - 0.25));
  expect_boundary(se_a_value, se_a_grad, kMax, locate_se_a_or_r(kMax));
  expect_linear_tail(se_a_value, se_a_grad, kMax + 0.25,
                     locate_se_a_or_r(kMax + 0.25));
}

TEST(TabulateExtrapolate, SeRUsesC1LinearTails) {
  expect_linear_tail(se_r_value, se_r_grad, kLower - 0.25,
                     locate_se_a_or_r(kLower - 0.25));
  expect_boundary(se_r_value, se_r_grad, kMax, locate_se_a_or_r(kMax));
  expect_linear_tail(se_r_value, se_r_grad, kMax + 0.25,
                     locate_se_a_or_r(kMax + 0.25));
}

TEST(TabulateExtrapolate, SeROffGridMaxUsesBoundarySegment) {
  const double max = kOffGridTableInfo[2];
  for (const double xx : {max, max + 0.25}) {
    const LocatedXx located = locate_se_a_or_r(xx, kOffGridTableInfo);
    EXPECT_NEAR(se_r_value(xx, kOffGridTableInfo),
                expected_table_value(located), kTolerance);
    EXPECT_NEAR(se_r_grad(xx, kOffGridTableInfo), expected_table_grad(located),
                kTolerance);
  }
}

TEST(TabulateExtrapolate, SeTUsesLinearLookupTails) {
  for (const double xx : {kMin - 0.25, kMin, kMax, kMax + 0.25}) {
    expect_se_t_lookup_tail(se_t_value, se_t_grad, xx);
  }
}

TEST(TabulateExtrapolate, SeTTebdUsesC1LinearTails) {
  expect_linear_tail(se_t_tebd_value, se_t_tebd_grad, kMin - 0.25,
                     locate_se_t(kMin - 0.25));
  expect_boundary(se_t_tebd_value, se_t_tebd_grad, kMax, locate_se_t(kMax));
  expect_linear_tail(se_t_tebd_value, se_t_tebd_grad, kMax + 0.25,
                     locate_se_t(kMax + 0.25));
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST(TabulateExtrapolate, SeAGpuUsesC1LinearTails) {
  expect_linear_tail(se_a_value_gpu, se_a_grad_gpu, kLower - 0.25,
                     locate_se_a_or_r(kLower - 0.25));
  expect_boundary(se_a_value_gpu, se_a_grad_gpu, kMax, locate_se_a_or_r(kMax));
  expect_linear_tail(se_a_value_gpu, se_a_grad_gpu, kMax + 0.25,
                     locate_se_a_or_r(kMax + 0.25));
}

TEST(TabulateExtrapolate, SeRGpuUsesC1LinearTails) {
  expect_linear_tail(se_r_value_gpu, se_r_grad_gpu, kLower - 0.25,
                     locate_se_a_or_r(kLower - 0.25));
  expect_boundary(se_r_value_gpu, se_r_grad_gpu, kMax, locate_se_a_or_r(kMax));
  expect_linear_tail(se_r_value_gpu, se_r_grad_gpu, kMax + 0.25,
                     locate_se_a_or_r(kMax + 0.25));
}

TEST(TabulateExtrapolate, SeTGpuUsesLinearLookupTails) {
  for (const double xx : {kMin - 0.25, kMin, kMax, kMax + 0.25}) {
    expect_se_t_lookup_tail(se_t_value_gpu, se_t_grad_gpu, xx);
  }
}

TEST(TabulateExtrapolate, SeTTebdGpuUsesC1LinearTails) {
  expect_linear_tail(se_t_tebd_value_gpu, se_t_tebd_grad_gpu, kMin - 0.25,
                     locate_se_t(kMin - 0.25));
  expect_boundary(se_t_tebd_value_gpu, se_t_tebd_grad_gpu, kMax,
                  locate_se_t(kMax));
  expect_linear_tail(se_t_tebd_value_gpu, se_t_tebd_grad_gpu, kMax + 0.25,
                     locate_se_t(kMax + 0.25));
}
#endif
