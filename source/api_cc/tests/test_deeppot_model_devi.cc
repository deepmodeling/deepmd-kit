#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include "DeepPot.h"
#include "neighbor_list.h"
#include "test_utils.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>  

class TestInferDeepPotModeDevi : public ::testing::Test
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
  int natoms;

  deepmd::DeepPot dp0;
  deepmd::DeepPot dp1;
  deepmd::DeepPotModelDevi dp_md;

  void SetUp() override {
    {
      std::string file_name = "../../tests/infer/deeppot.pbtxt";
      int fd = open(file_name.c_str(), O_RDONLY);
      tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
      tensorflow::GraphDef graph_def;
      tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
      delete input;
      std::fstream output("deeppot.pb", std::ios::out | std::ios::trunc | std::ios::binary);
      graph_def.SerializeToOstream(&output);
      dp0.init("deeppot.pb");
    }
    {
      std::string file_name = "../../tests/infer/deeppot-1.pbtxt";
      int fd = open(file_name.c_str(), O_RDONLY);
      tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
      tensorflow::GraphDef graph_def;
      tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
      delete input;
      std::fstream output("deeppot-1.pb", std::ios::out | std::ios::trunc | std::ios::binary);
      graph_def.SerializeToOstream(&output);
      dp1.init("deeppot-1.pb");
    }
    dp_md.init(std::vector<std::string>({"deeppot.pb", "deeppot-1.pb"}));
  };

  void TearDown() override {
    remove( "deeppot.pb" ) ;
    remove( "deeppot-1.pb" ) ;
  };
};

TEST_F(TestInferDeepPotModeDevi, attrs)
{
  EXPECT_EQ(dp0.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp0.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp0.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp0.dim_aparam(), dp_md.dim_aparam());
  EXPECT_EQ(dp1.cutoff(), dp_md.cutoff());
  EXPECT_EQ(dp1.numb_types(), dp_md.numb_types());
  EXPECT_EQ(dp1.dim_fparam(), dp_md.dim_fparam());
  EXPECT_EQ(dp1.dim_aparam(), dp_md.dim_aparam());
}


TEST_F(TestInferDeepPotModeDevi, cpu_lmp_list)
{
  float rc = dp_md.cutoff();
  int nloc = coord.size() / 3;  
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;  
  std::vector<std::vector<int > > nlist_data;
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
	       coord, atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);  

  int nmodel = 2;
  std::vector<double > edir(nmodel), emd;
  std::vector<std::vector<double> > fdir_(nmodel), fdir(nmodel), vdir(nmodel), fmd_, fmd(nmodel), vmd;
  dp0.compute(edir[0], fdir_[0], vdir[0], coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  for(int kk = 0; kk < nmodel; ++kk){
    _fold_back(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
  }
  

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  for(int kk = 0; kk < nmodel; ++kk){
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
  }  
  for(int kk = 0; kk < nmodel; ++kk){
    EXPECT_LT(fabs(edir[kk] - emd[kk]), 1e-10);
    for(int ii = 0; ii < fdir[0].size(); ++ii){
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), 1e-10);
    }
    for(int ii = 0; ii < vdir[0].size(); ++ii){
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), 1e-10);
    }
  }

  std::vector<double > avg_f, std_f;
  dp_md.compute_avg(avg_f, fmd);
  dp_md.compute_std_f(std_f, avg_f, fmd);
  
  // manual compute std f
  std::vector<double > manual_std_f(nloc);
  EXPECT_EQ(fmd[0].size(), nloc * 3);
  for(int ii = 0; ii < nloc; ++ii){
    std::vector<double > avg_f(3, 0.0);
    for(int dd = 0; dd < 3; ++dd){
      for(int kk = 0; kk < nmodel; ++kk){
	avg_f[dd] += fmd[kk][ii*3+dd];
      }
      avg_f[dd] /= (nmodel) * 1.0;
    }
    double std = 0.;
    for(int kk = 0; kk < nmodel; ++kk){
      for(int dd = 0; dd < 3; ++dd){
	double tmp = fmd[kk][ii*3+dd] - avg_f[dd];
	std += tmp * tmp;
      }
    }
    std /= nmodel * 1.0;
    manual_std_f[ii] = sqrt(std);
  }

  EXPECT_EQ(manual_std_f.size(), std_f.size());
  for(int ii = 0; ii < std_f.size(); ++ii){
    EXPECT_LT(fabs(manual_std_f[ii] - std_f[ii]), 1e-10);
  }
}


TEST_F(TestInferDeepPotModeDevi, cpu_lmp_list_atomic)
{
  float rc = dp_md.cutoff();
  int nloc = coord.size() / 3;  
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;  
  std::vector<std::vector<int > > nlist_data;
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
	       coord, atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);  

  int nmodel = 2;
  std::vector<double > edir(nmodel), emd;
  std::vector<std::vector<double> > fdir_(nmodel), fdir(nmodel), vdir(nmodel), fmd_, fmd(nmodel), vmd, aedir(nmodel), aemd, avdir(nmodel), avdir_(nmodel), avmd(nmodel), avmd_;
  dp0.compute(edir[0], fdir_[0], vdir[0], aedir[0], avdir_[0], coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  dp1.compute(edir[1], fdir_[1], vdir[1], aedir[1], avdir_[1], coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  dp_md.compute(emd, fmd_, vmd, aemd, avmd_, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  for(int kk = 0; kk < nmodel; ++kk){
    _fold_back(fdir[kk], fdir_[kk], mapping, nloc, nall, 3);
    _fold_back(fmd[kk], fmd_[kk], mapping, nloc, nall, 3);
    _fold_back(avdir[kk], avdir_[kk], mapping, nloc, nall, 9);
    _fold_back(avmd[kk], avmd_[kk], mapping, nloc, nall, 9);
  }  

  EXPECT_EQ(edir.size(), emd.size());
  EXPECT_EQ(fdir.size(), fmd.size());
  EXPECT_EQ(vdir.size(), vmd.size());
  EXPECT_EQ(aedir.size(), aemd.size());
  EXPECT_EQ(avdir.size(), avmd.size());
  for(int kk = 0; kk < nmodel; ++kk){
    EXPECT_EQ(fdir[kk].size(), fmd[kk].size());
    EXPECT_EQ(vdir[kk].size(), vmd[kk].size());
    EXPECT_EQ(aedir[kk].size(), aemd[kk].size());
    EXPECT_EQ(avdir[kk].size(), avmd[kk].size());
  }  
  for(int kk = 0; kk < nmodel; ++kk){
    EXPECT_LT(fabs(edir[kk] - emd[kk]), 1e-10);
    for(int ii = 0; ii < fdir[0].size(); ++ii){
      EXPECT_LT(fabs(fdir[kk][ii] - fmd[kk][ii]), 1e-10);
    }
    for(int ii = 0; ii < vdir[0].size(); ++ii){
      EXPECT_LT(fabs(vdir[kk][ii] - vmd[kk][ii]), 1e-10);
    }
    for(int ii = 0; ii < aedir[0].size(); ++ii){
      EXPECT_LT(fabs(aedir[kk][ii] - aemd[kk][ii]), 1e-10);
    }
    for(int ii = 0; ii < avdir[0].size(); ++ii){
      EXPECT_LT(fabs(avdir[kk][ii] - avmd[kk][ii]), 1e-10);
    }
  }
}

