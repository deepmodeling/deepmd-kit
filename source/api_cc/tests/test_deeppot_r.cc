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

class TestInferDeepPotR : public ::testing::Test
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
  std::vector<double> expected_e = {
    -9.320909762801588272e+01,-1.868020345400987878e+02,-1.868011172371355997e+02,-9.320868430396934912e+01,-1.868010398844378415e+02,-1.868016706555875999e+02
  };
  std::vector<double> expected_f = {
    6.385312846474267391e-04,-6.460452911141417731e-03,-5.652405655332678417e-04,-7.516468794343579736e-03,1.128804614240160216e-03,5.531937784564192051e-03,1.914138124904981664e-03,5.601819906021693503e-03,-5.131359585752605541e-03,-4.847104424804288617e-03,1.992071550328819614e-03,-4.028159855157302516e-03,1.236340684486603517e-03,-5.373955841338794344e-03,8.312829460571366513e-03,8.574563125108854156e-03,3.111712681889538742e-03,-4.120007238692381148e-03
  };
  std::vector<double> expected_v = {
    5.844056241889131371e-03,4.663973497239899614e-04,-2.268382127762904633e-03,4.663973497239897988e-04,2.349338784202595950e-03,-6.908546513234039253e-04,-2.268382127762904633e-03,-6.908546513234039253e-04,2.040499248150800561e-03,4.238130266437327605e-03,-1.539867187443782223e-04,-2.393101333240631613e-03,-1.539867187443782223e-04,4.410341945447907377e-04,9.544239698119633068e-06,-2.393101333240631613e-03,9.544239698119578858e-06,1.877785959095269654e-03,5.798992562057291543e-03,6.943392552230453693e-04,-1.180376879311998773e-03,6.943392552230453693e-04,1.686725132156275536e-03,-1.461632060145726542e-03,-1.180376879311998556e-03,-1.461632060145726325e-03,1.749543733794208444e-03,7.173915604192910439e-03,3.903218041111061569e-04,-5.747400467123527524e-04,3.903218041111061569e-04,1.208289706621179949e-03,-1.826828914132010932e-03,-5.747400467123527524e-04,-1.826828914132011148e-03,2.856960586657185906e-03,4.067553030177322240e-03,-3.267469855253819430e-05,-6.980667859103454904e-05,-3.267469855253830272e-05,1.387653029234650918e-03,-2.096820720698671855e-03,-6.980667859103444062e-05,-2.096820720698671855e-03,3.218305506720191278e-03,4.753992590355240674e-03,1.224911338353675992e-03,-1.683421934571502484e-03,1.224911338353676209e-03,7.332113564901583539e-04,-1.025577052190138451e-03,-1.683421934571502484e-03,-1.025577052190138234e-03,1.456681925652047018e-03
  };
  int natoms;
  double expected_tot_e;
  std::vector<double>expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deeppot-r.pbtxt";
    int fd = open(file_name.c_str(), O_RDONLY);
    tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
    tensorflow::GraphDef graph_def;
    tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    std::fstream output("deeppot.pb", std::ios::out | std::ios::trunc | std::ios::binary);
    graph_def.SerializeToOstream(&output);
    // check the string by the following commands
    // string txt;
    // tensorflow::protobuf::TextFormat::PrintToString(graph_def, &txt);

    dp.init("deeppot.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for(int ii = 0; ii < natoms; ++ii){
      expected_tot_e += expected_e[ii];
    }
    for(int ii = 0; ii < natoms; ++ii){
      for(int dd = 0; dd < 9; ++dd){
	expected_tot_v[dd] += expected_v[ii*9+dd];
      }
    }
  };

  void TearDown() override {
    remove( "deeppot.pb" ) ;
  };
};


TEST_F(TestInferDeepPotR, cpu_build_nlist)
{
  double ener;
  std::vector<double> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
}

TEST_F(TestInferDeepPotR, cpu_build_nlist_numfv)
{
  class MyModel : public EnergyModelTest<double>
  {
    deepmd::DeepPot & mydp;
    const std::vector<int > & atype;
public:
    MyModel(
	deepmd::DeepPot & dp_,
	const std::vector<int> & atype_
	) : mydp(dp_), atype(atype_) {};
    virtual void compute (
	double & ener,
	std::vector<double> &	force,
	std::vector<double> &	virial,
	const std::vector<double> & coord,
	const std::vector<double> & box) {
      mydp.compute(ener, force, virial, coord, atype, box);
    }
  };
  MyModel model(dp, atype);
  model.test_f(coord, box);
  model.test_v(coord, box);
  std::vector<double> box_(box);
  box_[1] -= 0.4;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[2] += 0.5;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[4] += 0.2;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[3] -= 0.3;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[6] -= 0.7;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
  box_[7] += 0.6;
  model.test_f(coord, box_);
  model.test_v(coord, box_);
}

TEST_F(TestInferDeepPotR, cpu_build_nlist_atomic)
{
  double ener;
  std::vector<double> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box);
  
  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms*9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms; ++ii){
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms*9; ++ii){
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), 1e-10);
  }
}


TEST_F(TestInferDeepPotR, cpu_lmp_nlist)
{
  float rc = dp.cutoff();
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
  
  double ener;
  std::vector<double> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  std::vector<double> force;
  _fold_back(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall-nloc, inlist, 1);
  _fold_back(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
}


TEST_F(TestInferDeepPotR, cpu_lmp_nlist_atomic)
{
  float rc = dp.cutoff();
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
  
  double ener;
  std::vector<double> force_, atom_ener_, atom_vir_, virial;
  std::vector<double> force, atom_ener, atom_vir;
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  _fold_back(force, force_, mapping, nloc, nall, 3);
  _fold_back(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back(atom_vir, atom_vir_, mapping, nloc, nall, 9);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms*9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms; ++ii){
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms*9; ++ii){
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), 1e-10);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(atom_ener_.begin(), atom_ener_.end(), 0.0);
  std::fill(atom_vir_.begin(), atom_vir_.end(), 0.0);  
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy, atype_cpy, box, nall-nloc, inlist, 1);
  _fold_back(force, force_, mapping, nloc, nall, 3);
  _fold_back(atom_ener, atom_ener_, mapping, nloc, nall, 1);
  _fold_back(atom_vir, atom_vir_, mapping, nloc, nall, 9);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_EQ(atom_vir.size(), natoms*9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms; ++ii){
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), 1e-10);
  }
  for(int ii = 0; ii < natoms*9; ++ii){
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), 1e-10);
  }
}


TEST_F(TestInferDeepPotR, cpu_lmp_nlist_2rc)
{
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;  
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;  
  std::vector<std::vector<int > > nlist_data;
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
	       coord, atype, box, rc*2);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);  
  
  double ener;
  std::vector<double> force_(nall*3, 0.0), virial(9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  std::vector<double> force;
  _fold_back(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }

  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall-nloc, inlist, 1);
  _fold_back(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
}


TEST_F(TestInferDeepPotR, cpu_lmp_nlist_type_sel)
{
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<double> coord_vir(nvir*3);
  std::vector<int> atype_vir(nvir, 2);
  for(int ii = 0; ii < nvir; ++ii){
    coord_vir[ii] = coord[ii];
  }  
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  natoms += nvir;
  std::vector<double> expected_f_vir(nvir*3, 0.0);
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(), expected_f_vir.end());

  // build nlist
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

  // dp compute
  double ener;
  std::vector<double> force_(nall*3, 0.0), virial(9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall-nloc, inlist, 0);
  // fold back
  std::vector<double> force;
  _fold_back(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
}



class TestInferDeepPotRNoPbc : public ::testing::Test
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
  std::vector<double> box = {};
  std::vector<double> expected_e = {
    -9.321213823508108476e+01,-1.868044102481340758e+02,-1.868067983858651075e+02,-9.320899631301440991e+01,-1.868014559732615112e+02,-1.868017660713088617e+02
  };
  std::vector<double> expected_f = {
    4.578151103701261042e-03,-1.917874111009987628e-03,-3.464546781179331930e-03,-4.578151103701261042e-03,1.917874111009987628e-03,3.464546781179331930e-03,-2.624402581721222913e-03,3.566275128489623933e-04,-2.859315986763691776e-04,-5.767787273464367384e-03,1.907053583551196647e-03,-3.889064429673861831e-03,1.786820066350549132e-04,-5.327197473636275694e-03,8.236236182834734409e-03,8.213507848550535492e-03,3.063516377236116545e-03,-4.061240154484504865e-03
  };
  std::vector<double> expected_v = {
            1.984979026299632174e-03,-8.315452677741701822e-04,-1.502146290172694243e-03,-8.315452677741700738e-04,3.483500446080982317e-04,6.292774999372096039e-04,-1.502146290172694243e-03,6.292774999372097123e-04,1.136759354725281907e-03,1.402852790439301908e-03,-5.876815743732210226e-04,-1.061618327900012114e-03,-5.876815743732211311e-04,2.461909298049979960e-04,4.447320022283834766e-04,-1.061618327900012331e-03,4.447320022283834766e-04,8.033868427351443728e-04,4.143606961846296385e-03,-5.511382161123719835e-04,4.465413399437045397e-04,-5.511382161123719835e-04,1.082271054025323839e-04,-1.097918001262628728e-04,4.465413399437046481e-04,-1.097918001262628728e-04,1.220966982358671871e-04,5.263952004497593831e-03,2.395243710938091842e-04,-2.830378939414603329e-04,2.395243710938094010e-04,1.189969706598244898e-03,-1.805627331015851201e-03,-2.830378939414602245e-04,-1.805627331015851635e-03,2.801996513751836820e-03,2.208413501170402270e-03,5.331756287635716889e-05,-1.664423506603235218e-04,5.331756287635695205e-05,1.379626072862918072e-03,-2.094132943741625064e-03,-1.664423506603234133e-04,-2.094132943741625064e-03,3.199787996743366607e-03,4.047014004814953811e-03,1.137904999421357000e-03,-1.568106936614101698e-03,1.137904999421357217e-03,7.205982843216952307e-04,-1.011174600268313238e-03,-1.568106936614101698e-03,-1.011174600268313238e-03,1.435226522157425754e-03
  };
  int natoms;
  double expected_tot_e;
  std::vector<double>expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deeppot-r.pbtxt";
    int fd = open(file_name.c_str(), O_RDONLY);
    tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
    tensorflow::GraphDef graph_def;
    tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    std::fstream output("deeppot.pb", std::ios::out | std::ios::trunc | std::ios::binary);
    graph_def.SerializeToOstream(&output);

    dp.init("deeppot.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for(int ii = 0; ii < natoms; ++ii){
      expected_tot_e += expected_e[ii];
    }
    for(int ii = 0; ii < natoms; ++ii){
      for(int dd = 0; dd < 9; ++dd){
	expected_tot_v[dd] += expected_v[ii*9+dd];
      }
    }
  };

  void TearDown() override {
    remove( "deeppot.pb" ) ;
  };
};

TEST_F(TestInferDeepPotRNoPbc, cpu_build_nlist)
{
  double ener;
  std::vector<double> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

  EXPECT_EQ(force.size(), natoms*3);
  EXPECT_EQ(virial.size(), 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for(int ii = 0; ii < natoms*3; ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);    
  }
  for(int ii = 0; ii < 3*3; ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
}
