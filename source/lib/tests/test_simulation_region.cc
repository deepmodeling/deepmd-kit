
#include <gtest/gtest.h>
#include <cmath>
#include "SimulationRegion.h"

double square_root (const double xx)
{
  return sqrt(xx);
}

TEST (SquareRootTest, PositiveNos) { 
    EXPECT_EQ (18.0, square_root (324.0));
    EXPECT_EQ (25.4, square_root (645.16));
    EXPECT_EQ (50.332, square_root (2533.310224));
}


