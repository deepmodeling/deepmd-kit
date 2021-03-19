#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>
#include "DeepTensor.h"
#include "DataModifier.h"
#include "SimulationRegion.h"
#include "ewald.h"
#include "neighbor_list.h"
#include "test_utils.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>  

class TestDipoleCharge : public ::testing::Test
{  
protected:  
  std::vector<double> coord = {
    4.6067455554,    8.8719311819,    6.3886531197,
    4.0044515745,    4.2449530507,    7.7902855220,
    2.6453069446,    0.8772647726,    1.2804446790,
    1.1445332290,    0.0067366438,    1.8606485070,
    7.1002867706,    5.0325506787,    3.1805888348,
    4.5352891138,    7.7389683929,    9.4260970128,
    2.1833238914,    9.0916071034,    7.2299906064,
    4.1040157820,    1.0496745045,    5.4748315591,
  };
  std::vector<int> atype = {
    0,3,2,1,3,4,1,4
  };
  std::vector<double> box = {
    10., 0., 0., 0., 10., 0., 0., 0., 10.
  };
  std::vector<double> expected_e = {
    3.671081837126222158e+00
  };
  std::vector<double> expected_f = {
    8.786854427753210128e-01,-1.590752486903602159e-01,-2.709225006303785932e-01,-4.449513960033193438e-01,-1.564291540964127813e-01,2.139031741772115178e-02,1.219699614140521193e+00,-5.580358618499958734e-02,-3.878662478349682585e-01,-1.286685244990778854e+00,1.886475802950296488e-01,3.904450515493615437e-01,1.605017382138404849e-02,2.138016869742287995e-01,-2.617514921203008965e-02,2.877081057057793712e-01,-3.846449683844421763e-01,3.048855616906603894e-02,-9.075632811311897807e-01,-6.509653472431625731e-03,2.302010972126376787e-01,2.370565856822822726e-01,3.600133435593881881e-01,1.243887532859055609e-02
  };
  std::vector<double> expected_v = {
    3.714071471995848417e-01,6.957130186032146613e-01,-1.158289779017217302e+00,6.957130186032139951e-01,-1.400130091653774933e+01,-3.631620234653316626e-01,-1.158289779017217302e+00,-3.631620234653316626e-01,3.805077486043773050e+00
  };
  std::vector<double> charge_map = {
    1., 1., 1., 1., 1., -1., -3.
  };
  int natoms;
  int ntypes;
  std::vector<int> type_asso;
  double expected_tot_e;
  std::vector<double>expected_tot_v;

  deepmd::DeepTensor dp;
  deepmd::DipoleChargeModifier dm;

  void SetUp() override {
    std::string file_name = "../../tests/infer/dipolecharge_e.pbtxt";
    int fd = open(file_name.c_str(), O_RDONLY);
    tensorflow::protobuf::io::ZeroCopyInputStream* input = new tensorflow::protobuf::io::FileInputStream(fd);
    tensorflow::GraphDef graph_def;
    tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
    delete input;
    std::string model = "dipolecharge_e.pb";
    std::fstream output(model.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    graph_def.SerializeToOstream(&output);
    // check the string by the following commands
    // string txt;
    // tensorflow::protobuf::TextFormat::PrintToString(graph_def, &txt);

    // dp.init("dipolecharge_d.pb");
    // dm.init("dipolecharge_d.pb");
    dp.init(model, 0, "dipole_charge");
    dm.init(model, 0, "dipole_charge");

    natoms = atype.size();
    ntypes = 5;
    type_asso.resize(ntypes, -1);
    type_asso[1] = 5;
    type_asso[3] = 6;

    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(9, expected_v.size());
  };

  void TearDown() override {
    remove( "dipolecharge_e.pb" ) ;
  };
};

static bool
_in_vec(const int & value,
	const std::vector<int> & vec)
{
  // naive impl.
  for(int ii = 0; ii < vec.size(); ++ii){
    if(value == vec[ii]) return true;
  }
  return false;
}

TEST_F(TestDipoleCharge, cpu_lmp_nlist)
{
  // build nlist
  // float rc = dp.cutoff();
  float rc = 4.0;
  int nloc = coord.size() / 3;  
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;  
  std::vector<std::vector<int > > nlist_data;
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
  	       coord, atype, box, rc);
  int nall = coord_cpy.size() / 3;
  int nghost = nall - nloc;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);  

  // evaluate dipole
  std::vector<double> dipole, dipole_recd(nloc*3, 0.0);
  dp.compute(dipole, coord_cpy, atype_cpy, box, nall-nloc, inlist);

  // add virtual atoms to the system
  // // a lot of mappings
  std::vector<int> sel_types = dp.sel_types();
  std::vector<int> sel_fwd, sel_bwd;
  int sel_nghost;
  deepmd::select_by_type(sel_fwd, sel_bwd, sel_nghost, coord_cpy, atype_cpy, nghost, sel_types);
  int sel_nall = sel_bwd.size();
  int sel_nloc = sel_nall - sel_nghost;
  std::vector<int> sel_atype(sel_bwd.size());
  deepmd::select_map<int>(sel_atype, atype, sel_fwd, 1);
  deepmd::AtomMap<double> nnp_map(sel_atype.begin(), sel_atype.begin() + sel_nloc);
  const std::vector<int> & sort_fwd_map(nnp_map.get_fwd_map());

  // // add coords
  std::vector<double > add_coord;
  std::vector<int > add_atype;
  std::vector<std::pair<int,int>> pairs;
  for(int ii = 0; ii < nloc; ++ii){
    if(_in_vec(atype[ii], sel_types)){
      int res_idx = sort_fwd_map[sel_fwd[ii]];
      std::vector<double > tmp_coord(3);
      for(int dd = 0; dd < 3; ++dd){
	tmp_coord[dd] = coord[ii*3+dd] + dipole[res_idx*3+dd];
	dipole_recd[ii*3+dd] = dipole[res_idx*3+dd];
      }
      pairs.push_back(std::pair<int,int>(ii, add_atype.size()+atype.size()));
      // std::cout << ii <<  " " 
      // 		<< atype[ii] << " " 
      // 		<< res_idx << " " 
      // 		<< type_asso[atype[ii]] << " " 
      // 		<< " pair "  
      // 		<< pairs.back().first << " " << pairs.back().second << " "
      // 		<< std::endl;
      add_coord.insert(add_coord.end(), tmp_coord.begin(), tmp_coord.end());
      add_atype.push_back(type_asso[atype[ii]]);      
    }
  }
  coord.insert(coord.end(), add_coord.begin(), add_coord.end());
  atype.insert(atype.end(), add_atype.begin(), add_atype.end());
  nloc = atype.size();
  EXPECT_EQ(atype.size()*3, coord.size());

  // get charge value
  std::vector<double> charge(nloc);
  for(int ii = 0; ii < nloc; ++ii){
    charge[ii] = charge_map[atype[ii]];
  }
  
  // compute the recp part of the ele interaction
  double eener;
  std::vector<double> eforce, evirial;
  deepmd::Region<double> region;
  init_region_cpu(region, &box[0]);
  deepmd::EwaldParameters<double> eparam;
  eparam.beta = 0.2;
  eparam.spacing = 4;
  ewald_recp(eener, eforce, evirial, coord, charge, region, eparam);
  
  EXPECT_LT(fabs(eener - expected_e[0]), 1e-6);
  EXPECT_EQ(eforce.size(), coord.size());
  EXPECT_EQ(evirial.size(), 9);  

  // extend the system with virtual atoms, and build nlist
  _build_nlist(nlist_data, coord_cpy, atype_cpy, mapping,
  	       coord, atype, box, rc);
  nall = coord_cpy.size() / 3;
  nghost = nall - nloc;
  ilist.resize(nloc);
  numneigh.resize(nloc);
  firstneigh.resize(nloc);
  inlist.inum = nloc;
  inlist.ilist = &ilist[0];
  inlist.numneigh = &numneigh[0];
  inlist.firstneigh = &firstneigh[0];
  convert_nlist(inlist, nlist_data);

  // compute force and virial
  std::vector<double > force_, force, virial;
  dm.compute(force_, virial, coord_cpy, atype_cpy, box, pairs, eforce, nghost, inlist);
  // for(int ii = 0; ii < force_.size(); ++ii){
  //   std::cout << force_[ii] << " " ;
  // }
  // std::cout << std::endl;
  _fold_back(force, force_, mapping, nloc, nall, 3);

  // compare force
  EXPECT_EQ(force.size(), nloc*3);
  // note nloc > expected_f.size(), because nloc contains virtual atoms.
  for(int ii = 0; ii < expected_f.size(); ++ii){
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-6);
  }

  // add recp virial and viral corr to virial
  // virial = virial_recp + virial_dipolecharge + virial_corr
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      virial[dd0*3+dd1] += evirial[dd0*3+dd1];
    }
  }    
  for(int ii = 0; ii < pairs.size(); ++ii){
    int idx0 = pairs[ii].first;
    int idx1 = pairs[ii].second;
    for (int dd0 = 0; dd0 < 3; ++dd0){
      for (int dd1 = 0; dd1 < 3; ++dd1){
	virial[dd0*3+dd1] -= eforce[idx1*3+dd0] * dipole_recd[idx0*3+dd1];
      }
    }    
  }
  // compare virial
  EXPECT_EQ(virial.size(), 3*3);
  for(int ii = 0; ii < expected_v.size(); ++ii){
    EXPECT_LT(fabs(virial[ii] - expected_v[ii]), 1e-5);
  }
}

