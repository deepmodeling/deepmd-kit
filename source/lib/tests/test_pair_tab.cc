#include <iostream>
#include <algorithm>
#include <gtest/gtest.h>
#include "fmt_nlist.h"
#include "env_mat.h"
#include "prod_env_mat.h"
#include "neighbor_list.h"
#include "pair_tab.h"

inline void
_cum_sum (
    std::vector<int> & sec,
    const std::vector<int> & n_sel) {
  sec.resize (n_sel.size() + 1);
  sec[0] = 0;
  for (int ii = 1; ii < sec.size(); ++ii){
    sec[ii] = sec[ii-1] + n_sel[ii-1];
  }
}

class TestPairTab : public ::testing::Test
{
protected:
  std::vector<double > posi = {
    12.83, 2.56, 2.18, 
    3.36, 3.00, 1.81,
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56
  };
  std::vector<double> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<int > atype = {0, 0, 1, 1, 1, 1};
  std::vector<double > posi_cpy;
  std::vector<int > atype_cpy;
  int ntypes = 2;  
  int nloc, nall, nnei, ndescrpt;
  std::vector<int> natoms;
  double rc = 6;
  double rc_smth = 0.8;
  SimulationRegion<double > region;
  std::vector<int> mapping, ncell, ngcell;
  std::vector<int> sel_a = {5, 5};
  std::vector<int> sel_r = {0, 0};  
  std::vector<int> sec_a, sec_r;
  // std::vector<int> sec_a = {0, 5, 10};
  // std::vector<int> sec_r = {0, 0, 0};
  std::vector<int> nat_stt, ext_stt, ext_end;
  std::vector<std::vector<int>> nlist_a_cpy, nlist_r_cpy;
  std::vector<double> env, env_deriv, rij;
  std::vector<int> nlist;
  std::vector<int> fmt_nlist_a;
  std::vector<double > tab_info = {
    0.000000000000000000e+00,5.000000000000000000e-01,1.900000000000000000e+01,2.000000000000000000e+00,
  };
  std::vector<double > tab_data = {
    -1.645731450145992980e-02,-9.318102695874852515e-03,5.052009558015377877e-01,0.000000000000000000e+00,-1.645731450146042940e-02,-5.869004620025458641e-02,4.371928069054082933e-01,4.794255386042030054e-01,-6.354779499724752534e-03,-1.080619897046354722e-01,2.704407710005182763e-01,8.414709848078965049e-01,3.676315333364321702e-03,-1.271263282038100351e-01,3.525245309207285238e-02,9.974949866040544455e-01,1.324335679744598204e-02,-1.160973822037167924e-01,-2.079712573154540167e-01,9.092974268256817094e-01,1.945112709784019289e-02,-7.636731181137940139e-02,-4.004359513305503215e-01,5.984721441039565493e-01,2.092788842815984651e-02,-1.801393051785876720e-02,-4.948171936597881015e-01,1.411200080598672135e-01,1.727238702609701360e-02,4.476973476662088336e-02,-4.680613894110260964e-01,-3.507832276896198365e-01,9.390240597413182511e-03,9.658689584491141067e-02,-3.267047587994936220e-01,-7.568024953079282025e-01,-7.915293177619131537e-04,1.247576176371514023e-01,-1.053602453174308090e-01,-9.775301176650970092e-01,-1.077948259470512538e-02,1.223830296838656073e-01,1.417804020035860202e-01,-9.589242746631384540e-01,-1.812776811548627576e-02,9.004458189975028670e-02,3.542080135872019420e-01,-7.055403255703919241e-01,-2.103966430679454769e-02,3.566127755329190352e-02,4.799138730402439101e-01,-2.794154981989258602e-01,-1.879310922837906794e-02,-2.745771536709157301e-02,4.881174352264442962e-01,2.151199880878155168e-01,-1.197225569894497244e-02,-8.383704305222891562e-02,3.768226768071237243e-01,6.569865987187890610e-01,-2.119743608124211032e-03,-1.197538101490639856e-01,1.732318236058309757e-01,9.379999767747388972e-01,7.876934490214426710e-03,-1.261130409734364521e-01,-7.263502751666940649e-02,9.893582466233817874e-01,1.734391611395907917e-02,-1.024822375027934496e-01,-3.012303059928992388e-01,7.984871126234902583e-01,1.734391611395902366e-02,-5.045048916091599001e-02,-4.541630326566090115e-01,4.121184852417565936e-01,-2.857038556912050442e-03,-1.088944917710427673e-01,3.641943791261976759e-01,7.071067811865474617e-01,-2.857038556912189220e-03,-1.174656074417785578e-01,1.378342799133762120e-01,9.595496299847904309e-01,9.997734062661967069e-03,-1.260367231125156806e-01,-1.056680506409179154e-01,9.770612638994756738e-01,1.714771792589186994e-02,-9.604352092452944634e-02,-3.277482946779627926e-01,7.553542242087043501e-01,2.097205155371145713e-02,-4.460036714685383652e-02,-4.683921827493462975e-01,3.487101265321038701e-01,1.942785366828164717e-02,1.831578751428042384e-02,-4.946767623819197657e-01,-1.433103718103848900e-01,1.318969794248442406e-02,7.659934851912514331e-02,-3.997616263485141985e-01,-6.002434930097426680e-01,3.705452789754024034e-03,1.161684423465786514e-01,-2.069938354828104177e-01,-9.102160728966471881e-01,-6.681509290541898238e-03,1.272848007158404737e-01,3.645940757960873524e-02,-9.973360132431250413e-01,-1.543384209711462507e-02,1.072402728442147790e-01,2.709844811396642239e-01,-8.402733142382176057e-01,-2.040699032626563936e-02,6.093874655287040421e-02,4.391635005367497957e-01,-4.774824023514532834e-01,-2.038432541044177260e-02,-2.822244259263473332e-04,4.998200226636939081e-01,2.212854411901365656e-03,-1.536918924167118838e-02,-6.143520065725210921e-02,4.381025975805153405e-01,4.813663272392268988e-01,-6.597319578834796860e-03,-1.075427683822652303e-01,2.691246285409980010e-01,8.426645349208191638e-01,3.812846748232762151e-03,-1.273347271187698360e-01,3.424713303996269192e-02,9.976490755007169087e-01,1.320350429809030723e-02,-1.158961868740716050e-01,-2.089837809528787005e-01,9.083743281701425198e-01,1.968240442794899625e-02,-7.628567397980012821e-02,-4.011656418067505725e-01,5.966978646412827159e-01,2.014467273961967342e-02,-1.723846069595319497e-02,-4.946897764825041177e-01,1.389289532826809004e-01,2.014467273961895177e-02,4.319555752290565875e-02,-4.687326796555513764e-01,-3.528546111561566834e-01,-2.857038556912050442e-03,-1.088944917710427673e-01,3.641943791261976759e-01,7.071067811865474617e-01,-2.857038556912189220e-03,-1.174656074417785578e-01,1.378342799133762120e-01,9.595496299847904309e-01,9.997734062661967069e-03,-1.260367231125156806e-01,-1.056680506409179154e-01,9.770612638994756738e-01,1.714771792589186994e-02,-9.604352092452944634e-02,-3.277482946779627926e-01,7.553542242087043501e-01,2.097205155371145713e-02,-4.460036714685383652e-02,-4.683921827493462975e-01,3.487101265321038701e-01,1.942785366828164717e-02,1.831578751428042384e-02,-4.946767623819197657e-01,-1.433103718103848900e-01,1.318969794248442406e-02,7.659934851912514331e-02,-3.997616263485141985e-01,-6.002434930097426680e-01,3.705452789754024034e-03,1.161684423465786514e-01,-2.069938354828104177e-01,-9.102160728966471881e-01,-6.681509290541898238e-03,1.272848007158404737e-01,3.645940757960873524e-02,-9.973360132431250413e-01,-1.543384209711462507e-02,1.072402728442147790e-01,2.709844811396642239e-01,-8.402733142382176057e-01,-2.040699032626563936e-02,6.093874655287040421e-02,4.391635005367497957e-01,-4.774824023514532834e-01,-2.038432541044177260e-02,-2.822244259263473332e-04,4.998200226636939081e-01,2.212854411901365656e-03,-1.536918924167118838e-02,-6.143520065725210921e-02,4.381025975805153405e-01,4.813663272392268988e-01,-6.597319578834796860e-03,-1.075427683822652303e-01,2.691246285409980010e-01,8.426645349208191638e-01,3.812846748232762151e-03,-1.273347271187698360e-01,3.424713303996269192e-02,9.976490755007169087e-01,1.320350429809030723e-02,-1.158961868740716050e-01,-2.089837809528787005e-01,9.083743281701425198e-01,1.968240442794899625e-02,-7.628567397980012821e-02,-4.011656418067505725e-01,5.966978646412827159e-01,2.014467273961967342e-02,-1.723846069595319497e-02,-4.946897764825041177e-01,1.389289532826809004e-01,2.014467273961895177e-02,4.319555752290565875e-02,-4.687326796555513764e-01,-3.528546111561566834e-01,1.241685182605223314e-02,-1.446819644344595757e-01,9.847674498780101260e-03,1.000000000000000000e+00,1.241685182605223314e-02,-1.074314089563028762e-01,-2.422656988919823506e-01,8.775825618903727587e-01,2.049371060414068024e-02,-7.018085347814612129e-02,-4.198779613264314037e-01,5.403023058681397650e-01,2.057421992118041443e-02,-8.699721665724191588e-03,-4.987585364703014945e-01,7.073720166770297579e-02,1.641560294060040448e-02,5.302293809781688516e-02,-4.544353200382087454e-01,-4.161468365471423514e-01,8.024007047643694213e-03,1.022697469196180153e-01,-2.991426350207735396e-01,-8.011436155469336962e-01,-2.274838714293875297e-03,1.263417680625494866e-01,-7.053112003860620427e-02,-9.899924966004454152e-01,-1.203208543609352033e-02,1.195172519196672223e-01,1.753278999436104768e-01,-9.364566872907963413e-01,-1.883932165321944296e-02,8.342099561138682784e-02,3.782661474746648045e-01,-6.536436208636119405e-01,-2.103521949550246628e-02,2.690303065172838792e-02,4.885901737377801313e-01,-2.107957994307797789e-01,-1.808035989191653092e-02,-3.620262783477895541e-02,4.792905765547292862e-01,2.836621854632261908e-01,-1.070002133978709136e-02,-9.044370751052860369e-02,3.526442412094217826e-01,7.086697742912599907e-01,-6.956515614555680571e-04,-1.225437715298900998e-01,1.396567621690030792e-01,9.601702866503659672e-01,9.463090404681517853e-03,-1.246307262142566930e-01,-1.075177355751436026e-01,9.765876257280234896e-01,1.736443528154563154e-02,-9.624145500021202837e-02,-3.283899167896124349e-01,7.539022543433047119e-01,2.079231845733497952e-02,-4.414814915557463415e-02,-4.687795209453995970e-01,3.466353178350258801e-01,1.995818879190347506e-02,1.822880621643041543e-02,-4.946988638845440933e-01,-1.455000338086134548e-01,1.114495328397846485e-02,7.810337259214095162e-02,-3.983666850759726152e-01,-6.020119026848235189e-01,1.114495328397829832e-02,1.115382324440758466e-01,-2.087250800397557615e-01,-9.111302618846769397e-01,
  };  
  std::vector<double> expected_energy = {
    -0.1306167788188060, -0.0255597250848064, 0.1587325724681873, -0.6817885971798407, -0.5510062343672764, 0.0991809936197377, 
  };
  
  void SetUp() override {
    do_setup();
  };
  void do_setup(){
    _cum_sum(sec_a, sel_a);
    _cum_sum(sec_r, sel_r);    
    region.reinitBox(&box[0]);
    copy_coord(posi_cpy, atype_cpy, mapping, ncell, ngcell, posi, atype, rc, region);
    nloc = posi.size() / 3;
    nall = posi_cpy.size() / 3;
    nnei = sec_a.back();
    ndescrpt = nnei * 4;
    natoms.resize(ntypes+2, 0);
    natoms[0] = nloc;
    natoms[1] = nall;
    for (int ii = 0; ii < nloc; ++ii){
      natoms[atype[ii]+2] ++;
    }
    nat_stt.resize(3);
    ext_stt.resize(3);
    ext_end.resize(3);
    for (int dd = 0; dd < 3; ++dd){
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_a_cpy, nlist_r_cpy, posi_cpy, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
    nlist.resize(nloc * nnei);
    env.resize(nloc * ndescrpt);
    env_deriv.resize(nloc * ndescrpt * 3);
    rij.resize(nloc * nnei * 3);
    for(int ii = 0; ii < nloc; ++ii){      
      // format nlist and record
      format_nlist_i_cpu<double>(fmt_nlist_a, posi_cpy, atype_cpy, ii, nlist_a_cpy[ii], rc, sec_a);
      for (int jj = 0; jj < nnei; ++jj){
	nlist[ii*nnei + jj] = fmt_nlist_a[jj];
      }
      std::vector<double > t_env, t_env_deriv, t_rij;
      // compute env_mat and its deriv, record
      deepmd::env_mat_a_cpu<double>(t_env, t_env_deriv, t_rij, posi_cpy, atype_cpy, ii, fmt_nlist_a, sec_a, rc_smth, rc);    
      for (int jj = 0; jj < ndescrpt; ++jj){
	env[ii*ndescrpt+jj] = t_env[jj];
	for (int dd = 0; dd < 3; ++dd){
	  env_deriv[ii*ndescrpt*3+jj*3+dd] = t_env_deriv[jj*3+dd];
	}
      }
      for (int jj = 0; jj < nnei * 3; ++jj){
	rij[ii*nnei*3 + jj] = t_rij[jj];
      }
    }
  }
  void TearDown() override {
  }
};


class TestPairTabTriBox : public TestPairTab
{
protected:  
  void SetUp() override {
    box = std::vector<double>({13., 0.3, 0.1, 0., 13., 0.2, 0., 0., 13.});
    do_setup();
  }
  void TearDown() override {
  }
};


TEST_F(TestPairTab, cpu)
{
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(nall * 9);
  std::vector<double > scale(nloc, 1.0);

  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  

  EXPECT_EQ(energy.size(), expected_energy.size());
  EXPECT_EQ(energy.size(), nloc);
  for (int ii = 0; ii < nloc; ++ii){
    EXPECT_LT(fabs(energy[ii] - expected_energy[ii]), 1e-8);
  }
  // for (int ii = 0; ii < nloc; ++ii){
  //   printf("%.16f, ", energy[ii]);    
  // }
  // printf("\n");
}


// int make_inter_nlist(
//     std::vector<int> &ilist,
//     std::vector<int> &jrange,
//     std::vector<int> &jlist,
//     const int & nloc,
//     const std::vector<std::vector<int>> & nlist_cpy)
// {
//   ilist.resize(nloc);
//   jrange.resize(nloc+1);
//   int tot_nnei = 0;
//   int max_nbor_size = 0;
//   for(int ii = 0; ii < nlist_cpy.size(); ++ii){
//     tot_nnei += nlist_cpy[ii].size();
//     if (nlist_cpy[ii].size() > max_nbor_size){
//       max_nbor_size = nlist_cpy[ii].size();
//     }
//   }
//   jlist.resize(tot_nnei);
//   for (int ii = 0; ii < nloc; ++ii){
//     ilist[ii] = ii;
//     jrange[ii+1] = jrange[ii] + nlist_cpy[ii].size();
//     int jj, cc;
//     for (jj = jrange[ii], cc = 0; jj < jrange[ii+1]; ++jj, ++cc){
//       jlist[jj] = nlist_cpy[ii][cc];
//     }
//   }
//   return max_nbor_size;
// }


TEST_F(TestPairTab, cpu_f_num_deriv)
{  
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(9, 0.);
  std::vector<double > atom_virial(nall * 9);
  std::vector<double > scale(nloc, 1.0);
  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &atom_virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  
  for (int ii = nloc; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      int orig_idx = mapping[ii];
      force[orig_idx*3+dd] += force[ii*3+dd];
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 9; ++dd){
      virial[dd] += atom_virial[ii*9+dd];
    }
  }
  double hh = 1e-4;
  for(int ii = 0; ii < nloc; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      std::vector<double> posi_0(posi);
      std::vector<double> posi_1(posi);
      posi_0[ii*3+dd] -= hh;
      posi_1[ii*3+dd] += hh;
      std::vector<double> posi_cpy_0, posi_cpy_1;
      std::vector<int> atype_cpy_0, atype_cpy_1;
      std::vector<int> t_mapping;
      copy_coord(posi_cpy_0, atype_cpy_0, t_mapping, ncell, ngcell, posi_0, atype, rc, region);
      copy_coord(posi_cpy_1, atype_cpy_1, t_mapping, ncell, ngcell, posi_1, atype, rc, region);
      EXPECT_EQ(atype_cpy_0, atype_cpy_1);
      for (int jj = 0; jj < atype_cpy_0.size(); ++jj){
	EXPECT_EQ(atype_cpy_0[jj], atype_cpy_1[jj]);
      }
      std::vector<std::vector<int>> nlist_cpy_0, nlist_cpy_1, t_nlist;
      build_nlist(nlist_cpy_0, t_nlist, posi_cpy_0, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      build_nlist(nlist_cpy_1, t_nlist, posi_cpy_1, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      std::vector<int> ilist_0(nloc), numneigh_0(nloc), ilist_1(nloc), numneigh_1(nloc);;
      std::vector<int*> firstneigh_0(nloc), firstneigh_1(nloc);
      deepmd::InputNlist inlist_0(nloc, &ilist_0[0], &numneigh_0[0], &firstneigh_0[0]);
      deepmd::InputNlist inlist_1(nloc, &ilist_1[0], &numneigh_1[0], &firstneigh_1[0]);
      convert_nlist(inlist_0, nlist_cpy_0);
      convert_nlist(inlist_1, nlist_cpy_1);
      int max_nnei_0 = max_numneigh(inlist_0);
      int max_nnei_1 = max_numneigh(inlist_1);
      EXPECT_EQ(max_nnei_0, max_nnei_1);
      std::vector<double> t_em(nloc * ndescrpt), t_em_deriv(nloc * ndescrpt * 3);
      std::vector<double> rij_0(nloc * nnei * 3), rij_1(nloc * nnei * 3);
      std::vector<int> nlist_0(nloc * nnei), nlist_1(nloc * nnei);
      std::vector<double > avg(ntypes * ndescrpt, 0);
      std::vector<double > std(ntypes * ndescrpt, 1);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_0[0], &nlist_0[0], &posi_cpy_0[0], &atype_cpy_0[0], inlist_0, max_nnei_0, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_1[0], &nlist_1[0], &posi_cpy_1[0], &atype_cpy_1[0], inlist_1, max_nnei_1, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      std::vector<double> energy_0(nloc), energy_1(nloc);
      std::vector<double> t_force(nall * 3), t_virial(nall * 9);
      deepmd::pair_tab_cpu(
	  &energy_0[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_0[0],
	  &scale[0],
	  &atype_cpy_0[0],
	  &nlist_0[0],
	  &natoms[0],
	  sel_a,
	  sel_r);  
      deepmd::pair_tab_cpu(
	  &energy_1[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_1[0],
	  &scale[0],
	  &atype_cpy_1[0],
	  &nlist_1[0],
	  &natoms[0],
	  sel_a,
	  sel_r); 
      double tot_e_0(0), tot_e_1(0);
      for(int ii = 0; ii < nloc; ++ii){
	tot_e_0 += energy_0[ii];
	tot_e_1 += energy_1[ii];
      }
      double num_deriv = - (tot_e_1 - tot_e_0) / (2. * hh);
      double ana_deriv = force[ii*3+dd];
      EXPECT_LT(fabs(num_deriv - ana_deriv), 1e-8);
    }  
  }
}


TEST_F(TestPairTab, cpu_f_num_deriv_scale)
{  
  double new_scale = 0.3;
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(9, 0.);
  std::vector<double > atom_virial(nall * 9);
  std::vector<double > scale(nloc, new_scale);
  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &atom_virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  
  for (int ii = nloc; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      int orig_idx = mapping[ii];
      force[orig_idx*3+dd] += force[ii*3+dd];
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 9; ++dd){
      virial[dd] += atom_virial[ii*9+dd];
    }
  }
  double hh = 1e-4;
  for(int ii = 0; ii < nloc; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      std::vector<double> posi_0(posi);
      std::vector<double> posi_1(posi);
      posi_0[ii*3+dd] -= hh;
      posi_1[ii*3+dd] += hh;
      std::vector<double> posi_cpy_0, posi_cpy_1;
      std::vector<int> atype_cpy_0, atype_cpy_1;
      std::vector<int> t_mapping;
      copy_coord(posi_cpy_0, atype_cpy_0, t_mapping, ncell, ngcell, posi_0, atype, rc, region);
      copy_coord(posi_cpy_1, atype_cpy_1, t_mapping, ncell, ngcell, posi_1, atype, rc, region);
      EXPECT_EQ(atype_cpy_0, atype_cpy_1);
      for (int jj = 0; jj < atype_cpy_0.size(); ++jj){
	EXPECT_EQ(atype_cpy_0[jj], atype_cpy_1[jj]);
      }
      std::vector<std::vector<int>> nlist_cpy_0, nlist_cpy_1, t_nlist;
      build_nlist(nlist_cpy_0, t_nlist, posi_cpy_0, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      build_nlist(nlist_cpy_1, t_nlist, posi_cpy_1, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      std::vector<int> ilist_0(nloc), numneigh_0(nloc), ilist_1(nloc), numneigh_1(nloc);;
      std::vector<int*> firstneigh_0(nloc), firstneigh_1(nloc);
      deepmd::InputNlist inlist_0(nloc, &ilist_0[0], &numneigh_0[0], &firstneigh_0[0]);
      deepmd::InputNlist inlist_1(nloc, &ilist_1[0], &numneigh_1[0], &firstneigh_1[0]);
      convert_nlist(inlist_0, nlist_cpy_0);
      convert_nlist(inlist_1, nlist_cpy_1);
      int max_nnei_0 = max_numneigh(inlist_0);
      int max_nnei_1 = max_numneigh(inlist_1);
      EXPECT_EQ(max_nnei_0, max_nnei_1);
      std::vector<double> t_em(nloc * ndescrpt), t_em_deriv(nloc * ndescrpt * 3);
      std::vector<double> rij_0(nloc * nnei * 3), rij_1(nloc * nnei * 3);
      std::vector<int> nlist_0(nloc * nnei), nlist_1(nloc * nnei);
      std::vector<double > avg(ntypes * ndescrpt, 0);
      std::vector<double > std(ntypes * ndescrpt, 1);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_0[0], &nlist_0[0], &posi_cpy_0[0], &atype_cpy_0[0], inlist_0, max_nnei_0, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_1[0], &nlist_1[0], &posi_cpy_1[0], &atype_cpy_1[0], inlist_1, max_nnei_1, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      std::vector<double> energy_0(nloc), energy_1(nloc);
      std::vector<double> t_force(nall * 3), t_virial(nall * 9);
      deepmd::pair_tab_cpu(
	  &energy_0[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_0[0],
	  &scale[0],
	  &atype_cpy_0[0],
	  &nlist_0[0],
	  &natoms[0],
	  sel_a,
	  sel_r);  
      deepmd::pair_tab_cpu(
	  &energy_1[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_1[0],
	  &scale[0],
	  &atype_cpy_1[0],
	  &nlist_1[0],
	  &natoms[0],
	  sel_a,
	  sel_r); 
      double tot_e_0(0), tot_e_1(0);
      for(int ii = 0; ii < nloc; ++ii){
	tot_e_0 += energy_0[ii];
	tot_e_1 += energy_1[ii];
      }
      double num_deriv = - (tot_e_1 - tot_e_0) / (2. * hh);
      double ana_deriv = force[ii*3+dd];
      EXPECT_LT(fabs(new_scale * num_deriv - ana_deriv), 1e-8);
    }  
  }
}

TEST_F(TestPairTab, cpu_v_num_deriv)
{
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(9, 0.);
  std::vector<double > atom_virial(nall * 9);
  std::vector<double > scale(nloc, 1.0);
  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &atom_virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  
  for (int ii = nloc; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      int orig_idx = mapping[ii];
      force[orig_idx*3+dd] += force[ii*3+dd];
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 9; ++dd){
      virial[dd] += atom_virial[ii*9+dd];
    }
  }
  double hh = 1e-4;
  std::vector<double> num_deriv(9);
  for(int dd0 = 0; dd0 < 3; ++dd0){
    for(int dd1 = 0; dd1 < 3; ++dd1){
      std::vector<double> box_0(9);
      std::vector<double> box_1(9);
      std::copy(box.begin(), box.end(), box_0.begin());
      std::copy(box.begin(), box.end(), box_1.begin());
      box_0[dd0*3+dd1] -= hh;
      box_1[dd0*3+dd1] += hh;
      SimulationRegion<double> region_0, region_1;
      region_0.reinitBox(&box_0[0]);
      region_1.reinitBox(&box_1[0]);
      std::vector<double> posi_0(nloc * 3), posi_1(nloc * 3);
      for(int jj = 0; jj < nloc; ++jj){
	double ci[3], co[3];
	region.phys2Inter(ci, &posi[jj*3]);
	region_0.inter2Phys(co, ci);
	std::copy(co, co+3, posi_0.begin() + jj*3);
	region_1.inter2Phys(co, ci);
	std::copy(co, co+3, posi_1.begin() + jj*3);	
      }
      std::vector<double> posi_cpy_0, posi_cpy_1;
      std::vector<int> atype_cpy_0, atype_cpy_1;
      std::vector<int> t_mapping;
      copy_coord(posi_cpy_0, atype_cpy_0, t_mapping, ncell, ngcell, posi_0, atype, rc, region_0);
      copy_coord(posi_cpy_1, atype_cpy_1, t_mapping, ncell, ngcell, posi_1, atype, rc, region_1);
      EXPECT_EQ(atype_cpy_0, atype_cpy_1);
      for (int jj = 0; jj < atype_cpy_0.size(); ++jj){
	EXPECT_EQ(atype_cpy_0[jj], atype_cpy_1[jj]);
      }
      std::vector<std::vector<int>> nlist_cpy_0, nlist_cpy_1, t_nlist;
      build_nlist(nlist_cpy_0, t_nlist, posi_cpy_0, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_0, ncell);
      build_nlist(nlist_cpy_1, t_nlist, posi_cpy_1, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_1, ncell);
      std::vector<int> ilist_0(nloc), numneigh_0(nloc), ilist_1(nloc), numneigh_1(nloc);;
      std::vector<int*> firstneigh_0(nloc), firstneigh_1(nloc);
      deepmd::InputNlist inlist_0(nloc, &ilist_0[0], &numneigh_0[0], &firstneigh_0[0]);
      deepmd::InputNlist inlist_1(nloc, &ilist_1[0], &numneigh_1[0], &firstneigh_1[0]);
      convert_nlist(inlist_0, nlist_cpy_0);
      convert_nlist(inlist_1, nlist_cpy_1);
      int max_nnei_0 = max_numneigh(inlist_0);
      int max_nnei_1 = max_numneigh(inlist_1);
      EXPECT_EQ(max_nnei_0, max_nnei_1);
      std::vector<double> t_em(nloc * ndescrpt), t_em_deriv(nloc * ndescrpt * 3);
      std::vector<double> rij_0(nloc * nnei * 3), rij_1(nloc * nnei * 3);
      std::vector<int> nlist_0(nloc * nnei), nlist_1(nloc * nnei);
      std::vector<double > avg(ntypes * ndescrpt, 0);
      std::vector<double > std(ntypes * ndescrpt, 1);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_0[0], &nlist_0[0], &posi_cpy_0[0], &atype_cpy_0[0], inlist_0, max_nnei_0, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_1[0], &nlist_1[0], &posi_cpy_1[0], &atype_cpy_1[0], inlist_1, max_nnei_1, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      std::vector<double> energy_0(nloc), energy_1(nloc);
      std::vector<double> t_force(nall * 3), t_virial(nall * 9);
      deepmd::pair_tab_cpu(
	  &energy_0[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_0[0],
	  &scale[0],
	  &atype_cpy_0[0],
	  &nlist_0[0],
	  &natoms[0],
	  sel_a,
	  sel_r);  
      deepmd::pair_tab_cpu(
	  &energy_1[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_1[0],
	  &scale[0],
	  &atype_cpy_1[0],
	  &nlist_1[0],
	  &natoms[0],
	  sel_a,
	  sel_r); 
      double tot_e_0(0), tot_e_1(0);
      for(int ii = 0; ii < nloc; ++ii){
	tot_e_0 += energy_0[ii];
	tot_e_1 += energy_1[ii];
      }
      num_deriv[dd0*3+dd1] = - (tot_e_1 - tot_e_0) / (2. * hh);
      // std::cout << num_deriv[dd0*3+dd1] << std::endl;
    }  
  }
  std::vector<double> num_vir(9, 0);
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      num_vir[dd0*3+dd1] = 0;
      for (int dd = 0; dd < 3; ++dd){
	num_vir[dd0*3+dd1] += num_deriv[dd*3+dd0] * box[dd*3+dd1];
      }
      // std::cout << num_vir[dd0*3+dd1] << " " << virial[dd0*3+dd1] << std::endl;
      EXPECT_LT(fabs(num_vir[dd0*3+dd1] - virial[dd0*3+dd1]), 1e-8);
    }
  }
}

TEST_F(TestPairTab, cpu_v_num_deriv_scale)
{
  double new_scale = 0.3;
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(9, 0.);
  std::vector<double > atom_virial(nall * 9);
  std::vector<double > scale(nloc, new_scale);
  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &atom_virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  
  for (int ii = nloc; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      int orig_idx = mapping[ii];
      force[orig_idx*3+dd] += force[ii*3+dd];
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 9; ++dd){
      virial[dd] += atom_virial[ii*9+dd];
    }
  }
  double hh = 1e-4;
  std::vector<double> num_deriv(9);
  for(int dd0 = 0; dd0 < 3; ++dd0){
    for(int dd1 = 0; dd1 < 3; ++dd1){
      std::vector<double> box_0(9);
      std::vector<double> box_1(9);
      std::copy(box.begin(), box.end(), box_0.begin());
      std::copy(box.begin(), box.end(), box_1.begin());
      box_0[dd0*3+dd1] -= hh;
      box_1[dd0*3+dd1] += hh;
      SimulationRegion<double> region_0, region_1;
      region_0.reinitBox(&box_0[0]);
      region_1.reinitBox(&box_1[0]);
      std::vector<double> posi_0(nloc * 3), posi_1(nloc * 3);
      for(int jj = 0; jj < nloc; ++jj){
	double ci[3], co[3];
	region.phys2Inter(ci, &posi[jj*3]);
	region_0.inter2Phys(co, ci);
	std::copy(co, co+3, posi_0.begin() + jj*3);
	region_1.inter2Phys(co, ci);
	std::copy(co, co+3, posi_1.begin() + jj*3);	
      }
      std::vector<double> posi_cpy_0, posi_cpy_1;
      std::vector<int> atype_cpy_0, atype_cpy_1;
      std::vector<int> t_mapping;
      copy_coord(posi_cpy_0, atype_cpy_0, t_mapping, ncell, ngcell, posi_0, atype, rc, region_0);
      copy_coord(posi_cpy_1, atype_cpy_1, t_mapping, ncell, ngcell, posi_1, atype, rc, region_1);
      EXPECT_EQ(atype_cpy_0, atype_cpy_1);
      for (int jj = 0; jj < atype_cpy_0.size(); ++jj){
	EXPECT_EQ(atype_cpy_0[jj], atype_cpy_1[jj]);
      }
      std::vector<std::vector<int>> nlist_cpy_0, nlist_cpy_1, t_nlist;
      build_nlist(nlist_cpy_0, t_nlist, posi_cpy_0, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_0, ncell);
      build_nlist(nlist_cpy_1, t_nlist, posi_cpy_1, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_1, ncell);
      std::vector<int> ilist_0(nloc), numneigh_0(nloc), ilist_1(nloc), numneigh_1(nloc);;
      std::vector<int*> firstneigh_0(nloc), firstneigh_1(nloc);
      deepmd::InputNlist inlist_0(nloc, &ilist_0[0], &numneigh_0[0], &firstneigh_0[0]);
      deepmd::InputNlist inlist_1(nloc, &ilist_1[0], &numneigh_1[0], &firstneigh_1[0]);
      convert_nlist(inlist_0, nlist_cpy_0);
      convert_nlist(inlist_1, nlist_cpy_1);
      int max_nnei_0 = max_numneigh(inlist_0);
      int max_nnei_1 = max_numneigh(inlist_1);
      EXPECT_EQ(max_nnei_0, max_nnei_1);
      std::vector<double> t_em(nloc * ndescrpt), t_em_deriv(nloc * ndescrpt * 3);
      std::vector<double> rij_0(nloc * nnei * 3), rij_1(nloc * nnei * 3);
      std::vector<int> nlist_0(nloc * nnei), nlist_1(nloc * nnei);
      std::vector<double > avg(ntypes * ndescrpt, 0);
      std::vector<double > std(ntypes * ndescrpt, 1);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_0[0], &nlist_0[0], &posi_cpy_0[0], &atype_cpy_0[0], inlist_0, max_nnei_0, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_1[0], &nlist_1[0], &posi_cpy_1[0], &atype_cpy_1[0], inlist_1, max_nnei_1, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      std::vector<double> energy_0(nloc), energy_1(nloc);
      std::vector<double> t_force(nall * 3), t_virial(nall * 9);
      deepmd::pair_tab_cpu(
	  &energy_0[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_0[0],
	  &scale[0],
	  &atype_cpy_0[0],
	  &nlist_0[0],
	  &natoms[0],
	  sel_a,
	  sel_r);  
      deepmd::pair_tab_cpu(
	  &energy_1[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_1[0],
	  &scale[0],
	  &atype_cpy_1[0],
	  &nlist_1[0],
	  &natoms[0],
	  sel_a,
	  sel_r); 
      double tot_e_0(0), tot_e_1(0);
      for(int ii = 0; ii < nloc; ++ii){
	tot_e_0 += energy_0[ii];
	tot_e_1 += energy_1[ii];
      }
      num_deriv[dd0*3+dd1] = - (tot_e_1 - tot_e_0) / (2. * hh);
      // std::cout << num_deriv[dd0*3+dd1] << std::endl;
    }  
  }
  std::vector<double> num_vir(9, 0);
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      num_vir[dd0*3+dd1] = 0;
      for (int dd = 0; dd < 3; ++dd){
	num_vir[dd0*3+dd1] += num_deriv[dd*3+dd0] * box[dd*3+dd1];
      }
      // std::cout << num_vir[dd0*3+dd1] << " " << virial[dd0*3+dd1] << std::endl;
      EXPECT_LT(fabs(new_scale * num_vir[dd0*3+dd1] - virial[dd0*3+dd1]), 1e-8);
    }
  }
}


TEST_F(TestPairTabTriBox, cpu_v_num_deriv)
{  
  std::vector<double > energy(nloc);
  std::vector<double > force(nall * 3);
  std::vector<double > virial(9, 0.);
  std::vector<double > atom_virial(nall * 9);
  std::vector<double > scale(nloc, 1.0);
  deepmd::pair_tab_cpu(
      &energy[0],
      &force[0],
      &atom_virial[0],
      &tab_info[0],
      &tab_data[0],
      &rij[0],
      &scale[0],
      &atype_cpy[0],
      &nlist[0],
      &natoms[0],
      sel_a, 
      sel_r);  
  for (int ii = nloc; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      int orig_idx = mapping[ii];
      force[orig_idx*3+dd] += force[ii*3+dd];
    }
  }
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 9; ++dd){
      virial[dd] += atom_virial[ii*9+dd];
    }
  }
  double hh = 1e-4;
  std::vector<double> num_deriv(9);
  for(int dd0 = 0; dd0 < 3; ++dd0){
    for(int dd1 = 0; dd1 < 3; ++dd1){
      std::vector<double> box_0(9);
      std::vector<double> box_1(9);
      std::copy(box.begin(), box.end(), box_0.begin());
      std::copy(box.begin(), box.end(), box_1.begin());
      box_0[dd0*3+dd1] -= hh;
      box_1[dd0*3+dd1] += hh;
      SimulationRegion<double> region_0, region_1;
      region_0.reinitBox(&box_0[0]);
      region_1.reinitBox(&box_1[0]);
      std::vector<double> posi_0(nloc * 3), posi_1(nloc * 3);
      for(int jj = 0; jj < nloc; ++jj){
	double ci[3], co[3];
	region.phys2Inter(ci, &posi[jj*3]);
	region_0.inter2Phys(co, ci);
	std::copy(co, co+3, posi_0.begin() + jj*3);
	region_1.inter2Phys(co, ci);
	std::copy(co, co+3, posi_1.begin() + jj*3);	
      }
      std::vector<double> posi_cpy_0, posi_cpy_1;
      std::vector<int> atype_cpy_0, atype_cpy_1;
      std::vector<int> t_mapping;
      copy_coord(posi_cpy_0, atype_cpy_0, t_mapping, ncell, ngcell, posi_0, atype, rc, region_0);
      copy_coord(posi_cpy_1, atype_cpy_1, t_mapping, ncell, ngcell, posi_1, atype, rc, region_1);
      EXPECT_EQ(atype_cpy_0, atype_cpy_1);
      for (int jj = 0; jj < atype_cpy_0.size(); ++jj){
	EXPECT_EQ(atype_cpy_0[jj], atype_cpy_1[jj]);
      }
      std::vector<std::vector<int>> nlist_cpy_0, nlist_cpy_1, t_nlist;
      build_nlist(nlist_cpy_0, t_nlist, posi_cpy_0, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_0, ncell);
      build_nlist(nlist_cpy_1, t_nlist, posi_cpy_1, nloc, rc, rc, nat_stt, ncell, ext_stt, ext_end, region_1, ncell);
      std::vector<int> ilist_0(nloc), numneigh_0(nloc), ilist_1(nloc), numneigh_1(nloc);;
      std::vector<int*> firstneigh_0(nloc), firstneigh_1(nloc);
      deepmd::InputNlist inlist_0(nloc, &ilist_0[0], &numneigh_0[0], &firstneigh_0[0]);
      deepmd::InputNlist inlist_1(nloc, &ilist_1[0], &numneigh_1[0], &firstneigh_1[0]);
      convert_nlist(inlist_0, nlist_cpy_0);
      convert_nlist(inlist_1, nlist_cpy_1);
      int max_nnei_0 = max_numneigh(inlist_0);
      int max_nnei_1 = max_numneigh(inlist_1);
      EXPECT_EQ(max_nnei_0, max_nnei_1);
      std::vector<double> t_em(nloc * ndescrpt), t_em_deriv(nloc * ndescrpt * 3);
      std::vector<double> rij_0(nloc * nnei * 3), rij_1(nloc * nnei * 3);
      std::vector<int> nlist_0(nloc * nnei), nlist_1(nloc * nnei);
      std::vector<double > avg(ntypes * ndescrpt, 0);
      std::vector<double > std(ntypes * ndescrpt, 1);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_0[0], &nlist_0[0], &posi_cpy_0[0], &atype_cpy_0[0], inlist_0, max_nnei_0, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      deepmd::prod_env_mat_a_cpu(&t_em[0], &t_em_deriv[0], &rij_1[0], &nlist_1[0], &posi_cpy_1[0], &atype_cpy_1[0], inlist_1, max_nnei_1, &avg[0], &std[0], nloc, nall, rc, rc_smth, sec_a);
      std::vector<double> energy_0(nloc), energy_1(nloc);
      std::vector<double> t_force(nall * 3), t_virial(nall * 9);
      deepmd::pair_tab_cpu(
	  &energy_0[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_0[0],
	  &scale[0],
	  &atype_cpy_0[0],
	  &nlist_0[0],
	  &natoms[0],
	  sel_a,
	  sel_r);  
      deepmd::pair_tab_cpu(
	  &energy_1[0],
	  &t_force[0],
	  &t_virial[0],
	  &tab_info[0],
	  &tab_data[0],
	  &rij_1[0],
	  &scale[0],
	  &atype_cpy_1[0],
	  &nlist_1[0],
	  &natoms[0],
	  sel_a,
	  sel_r); 
      double tot_e_0(0), tot_e_1(0);
      for(int ii = 0; ii < nloc; ++ii){
	tot_e_0 += energy_0[ii];
	tot_e_1 += energy_1[ii];
      }
      num_deriv[dd0*3+dd1] = - (tot_e_1 - tot_e_0) / (2. * hh);
      // std::cout << num_deriv[dd0*3+dd1] << std::endl;
    }  
  }
  std::vector<double> num_vir(9, 0);
  for (int dd0 = 0; dd0 < 3; ++dd0){
    for (int dd1 = 0; dd1 < 3; ++dd1){
      num_vir[dd0*3+dd1] = 0;
      for (int dd = 0; dd < 3; ++dd){
	num_vir[dd0*3+dd1] += num_deriv[dd*3+dd0] * box[dd*3+dd1];
      }
      // std::cout << num_vir[dd0*3+dd1] << " " << virial[dd0*3+dd1] << std::endl;
      EXPECT_LT(fabs(num_vir[dd0*3+dd1] - virial[dd0*3+dd1]), 1e-8);
    }
  }
}
