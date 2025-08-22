// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>

#include "common.h"

class TestPyTorchProfiler : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean any existing environment variables
    unsetenv("DP_ENABLE_PYTORCH_PROFILER");
    unsetenv("DP_PYTORCH_PROFILER_OUTPUT_DIR");
  }
  
  void TearDown() override {
    // Clean up environment variables
    unsetenv("DP_ENABLE_PYTORCH_PROFILER");
    unsetenv("DP_PYTORCH_PROFILER_OUTPUT_DIR");
  }
};

TEST_F(TestPyTorchProfiler, test_profiler_disabled_by_default) {
  bool enable_profiler;
  std::string output_dir;
  deepmd::get_env_pytorch_profiler(enable_profiler, output_dir);
  
  EXPECT_FALSE(enable_profiler);
  EXPECT_EQ(output_dir, "./profiler_output");
}

TEST_F(TestPyTorchProfiler, test_profiler_enabled_with_env) {
  setenv("DP_ENABLE_PYTORCH_PROFILER", "1", 1);
  
  bool enable_profiler;
  std::string output_dir;
  deepmd::get_env_pytorch_profiler(enable_profiler, output_dir);
  
  EXPECT_TRUE(enable_profiler);
  EXPECT_EQ(output_dir, "./profiler_output");
}

TEST_F(TestPyTorchProfiler, test_profiler_enabled_with_true) {
  setenv("DP_ENABLE_PYTORCH_PROFILER", "true", 1);
  
  bool enable_profiler;
  std::string output_dir;
  deepmd::get_env_pytorch_profiler(enable_profiler, output_dir);
  
  EXPECT_TRUE(enable_profiler);
  EXPECT_EQ(output_dir, "./profiler_output");
}

TEST_F(TestPyTorchProfiler, test_custom_output_dir) {
  setenv("DP_ENABLE_PYTORCH_PROFILER", "1", 1);
  setenv("DP_PYTORCH_PROFILER_OUTPUT_DIR", "/custom/path", 1);
  
  bool enable_profiler;
  std::string output_dir;
  deepmd::get_env_pytorch_profiler(enable_profiler, output_dir);
  
  EXPECT_TRUE(enable_profiler);
  EXPECT_EQ(output_dir, "/custom/path");
}

TEST_F(TestPyTorchProfiler, test_profiler_disabled_with_zero) {
  setenv("DP_ENABLE_PYTORCH_PROFILER", "0", 1);
  
  bool enable_profiler;
  std::string output_dir;
  deepmd::get_env_pytorch_profiler(enable_profiler, output_dir);
  
  EXPECT_FALSE(enable_profiler);
  EXPECT_EQ(output_dir, "./profiler_output");
}

TEST_F(TestPyTorchProfiler, test_mpi_rank_detection) {
  // Test that MPI rank detection returns valid rank (-1 when MPI not initialized, >= 0 when initialized)
  int rank = deepmd::get_mpi_rank();
  EXPECT_GE(rank, -1);  // Rank should be -1 (not available) or >= 0 (valid rank)
}