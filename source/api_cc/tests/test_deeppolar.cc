#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include "DeepTensor.h"
#include "neighbor_list.h"
#include "test_utils.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>  

class TestInferDeepPolar : public ::testing::Test
{  
protected:  
  std::vector<double> coord = {
    12.83, 2.56, 2.18,
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.36, 3.00, 1.81,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56 
  };
  std::vector<int> atype = {
    0, 1, 1, 0, 1, 1
  };
  std::vector<double> box = {
    13., 0., 0., 0., 13., 0., 0., 0., 13.
  };
  std::vector<double> expected_d = {
    1.061407927405987051e-01,-3.569013342133873778e-01,-2.862108976089940138e-02,-3.569013342133875444e-01,1.304367268874677244e+00,1.037647501453442256e-01,-2.862108976089940138e-02,1.037647501453441284e-01,8.100521520762453409e-03,1.236797829492216616e+00,-3.717307430531632262e-01,7.371515676976750919e-01,-3.717307430531630041e-01,1.127222682121889058e-01,-2.239181552775717510e-01,7.371515676976746478e-01,-2.239181552775717787e-01,4.448255365635306879e-01
  };
  int natoms;

  deepmd::DeepTensor dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deeppolar.pbtxt";
    int fd = open(file_name.c_str(), O_RDONLY);
    tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
    tensorflow::GraphDef graph_def;
    tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    std::fstream output("deeppolar.pb", std::ios::out | std::ios::trunc | std::ios::binary);
    graph_def.SerializeToOstream(&output);
    // check the string by the following commands
    // string txt;
    // tensorflow::protobuf::TextFormat::PrintToString(graph_def, &txt);

    dp.init("deeppolar.pb");

    natoms = expected_d.size();
  };

  void TearDown() override {
    remove( "deeppolar.pb" ) ;
  };
};


TEST_F(TestInferDeepPolar, cpu_build_nlist)
{
  EXPECT_EQ(dp.cutoff(), 6.);
  EXPECT_EQ(dp.numb_types(), 2);
  EXPECT_EQ(dp.output_dim(), 9);
  std::vector<int> sel_types = dp.sel_types();
  EXPECT_EQ(sel_types.size(), 1);
  EXPECT_EQ(sel_types[0], 0);

  std::vector<double> value;
  dp.compute(value, coord, atype, box);

  EXPECT_EQ(value.size(), expected_d.size());
  for(int ii = 0; ii < expected_d.size(); ++ii){
    EXPECT_LT(fabs(value[ii] - expected_d[ii]), 1e-10);
  }
}

TEST_F(TestInferDeepPolar, cpu_lmp_nlist)
{
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;  
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;  
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  std::vector<std::vector<int > > nlist_data;
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
	       coord, atype, box, rc);
  int nall = coord_cpy.size() / 3;
  convert_nlist(inlist, nlist_data);  

  std::vector<double> value;
  dp.compute(value, coord_cpy, atype_cpy, box, nall-nloc, inlist);

  EXPECT_EQ(value.size(), expected_d.size());
  for(int ii = 0; ii < expected_d.size(); ++ii){
    EXPECT_LT(fabs(value[ii] - expected_d[ii]), 1e-10);
  }
}

