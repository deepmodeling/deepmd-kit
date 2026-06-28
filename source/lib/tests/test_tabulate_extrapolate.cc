// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

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

LocatedXx locate_se_a_or_r(const double xx) {
  if (xx < kLower) {
    return {0, 0.0, xx - kLower};
  }
  if (xx < kUpper) {
    const int table_idx = static_cast<int>((xx - kLower) / kStride0);
    return {table_idx, xx - (table_idx * kStride0 + kLower), 0.0};
  }
  if (xx < kMax) {
    const int first_stride = static_cast<int>((kUpper - kLower) / kStride0);
    const int table_idx =
        first_stride + static_cast<int>((xx - kUpper) / kStride1);
    return {table_idx, xx - ((table_idx - first_stride) * kStride1 + kUpper),
            0.0};
  }

  const int first_stride = static_cast<int>((kUpper - kLower) / kStride0);
  const int table_idx =
      first_stride + static_cast<int>((kMax - kUpper) / kStride1) - 1;
  return {table_idx, kMax - ((table_idx - first_stride) * kStride1 + kUpper),
          xx - kMax};
}

LocatedXx locate_se_t(const double xx) {
  if (xx < kMin) {
    return {0, 0.0, xx - kMin};
  }
  if (xx < kLower) {
    const int table_idx = static_cast<int>((xx - kMin) / kStride1);
    return {table_idx, xx - (table_idx * kStride1 + kMin), 0.0};
  }
  if (xx < kUpper) {
    const int first_stride = static_cast<int>((kLower - kMin) / kStride1);
    const int table_idx =
        first_stride + static_cast<int>((xx - kLower) / kStride0);
    return {table_idx, xx - ((table_idx - first_stride) * kStride0 + kLower),
            0.0};
  }
  if (xx < kMax) {
    const int first_stride = static_cast<int>((kLower - kMin) / kStride1) +
                             static_cast<int>((kUpper - kLower) / kStride0);
    const int table_idx =
        first_stride + static_cast<int>((xx - kUpper) / kStride1);
    return {table_idx, xx - ((table_idx - first_stride) * kStride1 + kUpper),
            0.0};
  }

  const int first_stride = static_cast<int>((kLower - kMin) / kStride1) +
                           static_cast<int>((kUpper - kLower) / kStride0);
  const int table_idx =
      first_stride + static_cast<int>((kMax - kUpper) / kStride1) - 1;
  return {table_idx, kMax - ((table_idx - first_stride) * kStride1 + kUpper),
          xx - kMax};
}

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

double se_r_value(const double xx) {
  std::vector<double> out(1);
  const std::vector<double> em = {xx};
  deepmd::tabulate_fusion_se_r_cpu(out.data(), kTable.data(), kTableInfo.data(),
                                   em.data(), 1, 1, kLastLayerSize);
  return out[0];
}

double se_r_grad(const double xx) {
  std::vector<double> dy_dem(1);
  const std::vector<double> em = {xx};
  const std::vector<double> dy = {1.0};
  deepmd::tabulate_fusion_se_r_grad_cpu(dy_dem.data(), kTable.data(),
                                        kTableInfo.data(), em.data(), dy.data(),
                                        1, 1, kLastLayerSize);
  return dy_dem[0];
}

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

TEST(TabulateExtrapolate, SeTUsesLinearLookupTails) {
  for (const double xx : {kMin - 0.25, kMax, kMax + 0.25}) {
    const LocatedXx located = locate_se_t(xx);
    const double expected_value = expected_table_value(located);
    const double expected_grad = expected_table_grad(located);
    double dy_dem_x = 0.0;
    double dy_dem = 0.0;
    se_t_grad(xx, dy_dem_x, dy_dem);
    EXPECT_NEAR(se_t_value(xx), xx * expected_value, kTolerance);
    EXPECT_NEAR(dy_dem_x, xx * expected_grad, kTolerance);
    EXPECT_NEAR(dy_dem, expected_value, kTolerance);
  }
}

TEST(TabulateExtrapolate, SeTTebdUsesC1LinearTails) {
  expect_linear_tail(se_t_tebd_value, se_t_tebd_grad, kMin - 0.25,
                     locate_se_t(kMin - 0.25));
  expect_boundary(se_t_tebd_value, se_t_tebd_grad, kMax, locate_se_t(kMax));
  expect_linear_tail(se_t_tebd_value, se_t_tebd_grad, kMax + 0.25,
                     locate_se_t(kMax + 0.25));
}
