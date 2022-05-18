#include "PaddleDeepPot.h"
#include "AtomMap.h"
#include <stdexcept>	

using namespace deepmd;

static 
std::vector<int> cum_sum (const std::vector<int> & n_sel) {
    std::vector<int> sec;
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii) {
        sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
    return sec;
}

// o_energy: save_infer_model/scale_4.tmp_1
// o_force: save_infer_model/scale_5.tmp_1
// o_atom_virial.tmp_0: save_infer_model/scale_1.tmp_1
static void 
paddle_run_model(ENERGYTYPE &	dener,
	   std::vector<VALUETYPE> &	dforce_,
	   std::vector<VALUETYPE> &	dvirial,
	   std::shared_ptr<paddle_infer::Predictor> predictor,
	   const AtomMap<VALUETYPE>&	atommap, 
	   const int			nghost = 0)
{
    unsigned nloc = atommap.get_type().size();
    unsigned nall = nloc + nghost;
    if (nloc == 0) {
        dener = 0;
        // no backward map needed
        // dforce of size nall * 3
        dforce_.resize(nall * 3);
        fill(dforce_.begin(), dforce_.end(), 0.0);
        // dvirial of size 9
        dvirial.resize(9);
        fill(dvirial.begin(), dvirial.end(), 0.0);
        return;
    }
    predictor->Run();
    // Deal with output
    // std::vector<deepmd::ENERGYTYPE> oe;
    std::vector<float> oe;
    std::vector<VALUETYPE> of, oav;
    auto output_names = predictor->GetOutputNames();

    auto out_e_t = predictor->GetOutputHandle(output_names[4]);
    std::vector<int> out_e_shape = out_e_t->shape();
    int out_e_num = std::accumulate(out_e_shape.begin(), out_e_shape.end(), 1,
                                    std::multiplies<int>());
    oe.resize(out_e_num);
    out_e_t->CopyToCpu(oe.data());

    auto out_f_t = predictor->GetOutputHandle(output_names[5]);
    std::vector<int> out_f_shape = out_f_t->shape();
    int out_f_num = std::accumulate(out_f_shape.begin(), out_f_shape.end(), 1,
                                    std::multiplies<int>());
    of.resize(out_f_num);
    out_f_t->CopyToCpu(of.data());

    auto out_v_t = predictor->GetOutputHandle(output_names[1]);
    std::vector<int> out_v_shape = out_v_t->shape();
    int out_v_num = std::accumulate(out_v_shape.begin(), out_v_shape.end(), 1,
                                    std::multiplies<int>());
    oav.resize(out_v_num);
    out_v_t->CopyToCpu(oav.data());

    dener = static_cast<deepmd::ENERGYTYPE>(oe[0]);
    std::vector<VALUETYPE> dforce (3 * nall, 0.0);
    dvirial.resize(9);
    for (int i = 0; i < nall * 3; ++i) {
        dforce[i] = of[i];
    }
    for (int i = 0; i < nall; ++i) {
        dvirial[0] += 1.0 * oav[9*i+0];
        dvirial[1] += 1.0 * oav[9*i+1];
        dvirial[2] += 1.0 * oav[9*i+2];
        dvirial[3] += 1.0 * oav[9*i+3];
        dvirial[4] += 1.0 * oav[9*i+4];
        dvirial[5] += 1.0 * oav[9*i+5];
        dvirial[6] += 1.0 * oav[9*i+6];
        dvirial[7] += 1.0 * oav[9*i+7];
        dvirial[8] += 1.0 * oav[9*i+8];
	  }
    dforce_ = dforce;
    atommap.backward (dforce_.begin(), dforce.begin(), 3);
}

// o_energy" save_infer_model/scale_4.tmp_1
// "o_force" save_infer_model/scale_5.tmp_1
// "o_atom_energy" save_infer_model/scale_0.tmp_1
// "o_atom_virial" save_infer_model/scale_1.tmp_1

static void 
paddle_run_model (deepmd::ENERGYTYPE   &		dener,
		       std::vector<deepmd::VALUETYPE>&	dforce_,
		       std::vector<deepmd::VALUETYPE>&	dvirial,	   
		       std::vector<deepmd::VALUETYPE>&	datom_energy_,
		       std::vector<deepmd::VALUETYPE>&	datom_virial_,
		       std::shared_ptr<paddle_infer::Predictor> predictor,
		       const deepmd::AtomMap<VALUETYPE> &   atommap, 
		       const int&		nghost = 0){
    unsigned nloc = atommap.get_type().size();
    unsigned nall = nloc + nghost;
    if (nloc == 0) {
        dener = 0;
        // no backward map needed
        // dforce of size nall * 3
        dforce_.resize(nall * 3);
        fill(dforce_.begin(), dforce_.end(), 0.0);
        // dvirial of size 9
        dvirial.resize(9);
        fill(dvirial.begin(), dvirial.end(), 0.0);
        // datom_energy_ of size nall
        datom_energy_.resize(nall);
        fill(datom_energy_.begin(), datom_energy_.end(), 0.0);
        // datom_virial_ of size nall * 9
        datom_virial_.resize(nall * 9);
        fill(datom_virial_.begin(), datom_virial_.end(), 0.0);
        return;
    }
    predictor->Run();
    // {"o_energy", "o_force", "o_atom_energy", "o_atom_virial"},
    // Deal with output
    // std::vector<deepmd::ENERGYTYPE> oe;
    std::vector<float> oe;
    std::vector<deepmd::VALUETYPE> of, oav, oae;
    auto output_names = predictor->GetOutputNames();

    auto out_e_t = predictor->GetOutputHandle(output_names[4]);
    std::vector<int> out_e_shape = out_e_t->shape();
    int out_e_num = std::accumulate(out_e_shape.begin(), out_e_shape.end(), 1,
                                    std::multiplies<int>());
    oe.resize(out_e_num);
    out_e_t->CopyToCpu(oe.data());

    auto out_f_t = predictor->GetOutputHandle(output_names[5]);
    std::vector<int> out_f_shape = out_f_t->shape();
    int out_f_num = std::accumulate(out_f_shape.begin(), out_f_shape.end(), 1,
                                    std::multiplies<int>());
    of.resize(out_f_num);
    out_f_t->CopyToCpu(of.data());

    auto out_ae_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> out_ae_shape = out_ae_t->shape();
    int out_ae_num = std::accumulate(out_ae_shape.begin(), out_ae_shape.end(), 1,
                                    std::multiplies<int>());
    oae.resize(out_ae_num);
    out_ae_t->CopyToCpu(oae.data());

    auto out_av_t = predictor->GetOutputHandle(output_names[1]);
    std::vector<int> out_av_shape = out_av_t->shape();
    int out_av_num = std::accumulate(out_av_shape.begin(), out_av_shape.end(), 1,
                                    std::multiplies<int>());
    oav.resize(out_av_num);
    out_av_t->CopyToCpu(oav.data());

    dener = static_cast<deepmd::ENERGYTYPE>(oe[0]);
    std::vector<VALUETYPE> dforce (3 * nall);
    std::vector<VALUETYPE> datom_energy (nall, 0);
    std::vector<VALUETYPE> datom_virial (9 * nall);
    dvirial.resize (9);
    for (int ii = 0; ii < nall * 3; ++ii) {
        dforce[ii] = of[ii];
    }
    for (int ii = 0; ii < nloc; ++ii) {
        datom_energy[ii] = oae[ii];
    }
    for (int ii = 0; ii < nall * 9; ++ii) {
        datom_virial[ii] = oav[ii];
    }
    for (int ii = 0; ii < nall; ++ii) {
        dvirial[0] += 1.0 * datom_virial[9*ii+0];
        dvirial[1] += 1.0 * datom_virial[9*ii+1];
        dvirial[2] += 1.0 * datom_virial[9*ii+2];
        dvirial[3] += 1.0 * datom_virial[9*ii+3];
        dvirial[4] += 1.0 * datom_virial[9*ii+4];
        dvirial[5] += 1.0 * datom_virial[9*ii+5];
        dvirial[6] += 1.0 * datom_virial[9*ii+6];
        dvirial[7] += 1.0 * datom_virial[9*ii+7];
        dvirial[8] += 1.0 * datom_virial[9*ii+8];
	}
    dforce_ = dforce;
    datom_energy_ = datom_energy;
    datom_virial_ = datom_virial;
    atommap.backward (dforce_.begin(), dforce.begin(), 3);
    atommap.backward (datom_energy_.begin(), datom_energy.begin(), 1);
    atommap.backward (datom_virial_.begin(), datom_virial.begin(), 9);
}

PaddleDeepPot::PaddleDeepPot (): inited (false),init_nbor (false)
{

}

PaddleDeepPot::PaddleDeepPot (std::string& prog_str, std::string& params_str)
    : inited (false), init_nbor (false)
{
  init(prog_str, params_str);  
}

PaddleDeepPot::~PaddleDeepPot() {}

void PaddleDeepPot::init (std::string& prog_str, std::string& params_str)
{
  if (inited){
    std::cerr << "Init error" << std::endl;
    return ;
  }
  math_lib_num_threads = deepmd::get_math_lib_num_threads();
  config.SetModelBuffer(prog_str.c_str(), prog_str.size(), params_str.c_str(), params_str.size());
  // config.SwitchIrOptim();
  // config.EnableMKLDNN();
  config.SetCpuMathLibraryNumThreads(math_lib_num_threads);
  std::cout<<">>> Info of cpu math lib num threads: "<< deepmd::get_math_lib_num_threads() <<std::endl;
  predictor = paddle_infer::CreatePredictor(config);

/*
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut")
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam_ = get_scalar<int>("fitting_attr/dfparam");
  daparam_ = get_scalar<int>("fitting_attr/daparam");
  dfparam_ = dfparam_ < 0 ? 0 : dfparam_;
  daparam_ = daparam_ < 0 ? 0 : daparam_;
*/
  cell_size = rcut;
  dfparam = 0;
  daparam = 0;
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;

  inited = true;
  init_nbor = false;
}

void 
PaddleDeepPot::
print_summary(const std::string &pre) const
{
  std::cout << pre << "installed to:       " + global_install_prefix << std::endl;
  std::cout << pre << "source:             " + global_git_summ << std::endl;
  std::cout << pre << "source brach:       " + global_git_branch << std::endl;
  std::cout << pre << "source commit:      " + global_git_hash << std::endl;
  std::cout << pre << "source commit at:   " + global_git_date << std::endl;
  std::cout << pre << "surpport model ver.:" + global_model_version << std::endl;
  std::cout << pre << "build float prec:   " + global_float_prec << std::endl;
  std::cout << pre << "build with paddle inference framework " << std::endl;
}

void 
PaddleDeepPot::
validate_fparam_aparam(const int & nloc,
		       const std::vector<VALUETYPE> &fparam,
		       const std::vector<VALUETYPE> &aparam, 
           const int &dfparam, 
           const int &daparam) const
{
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }
  if (aparam.size() != daparam * nloc) {
    throw std::runtime_error("the dim of atom parameter provided is not consistent with what the model uses");
  }  
}

void PaddleDeepPot::compute (ENERGYTYPE & dener,
    std::vector<VALUETYPE> &	dforce_,
    std::vector<VALUETYPE> &	dvirial,
    const std::vector<VALUETYPE> &	dcoord_,
    const std::vector<int> &	datype_,
    const std::vector<VALUETYPE> &	dbox,
    const std::vector<VALUETYPE>&	fparam,
    const std::vector<VALUETYPE>&	aparam){
    
    // Initialization
    int nall = dcoord_.size() / 3;
    int nloc = nall;
    atommap = deepmd::AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
    assert (nloc == atommap.get_type().size());
    validate_fparam_aparam(nloc, fparam, aparam, dfparam, daparam);

    int ret = paddle_input_tensors(predictor, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam, atommap);
    // Execution
    assert (ret == nloc);

    paddle_run_model(dener, dforce_, dvirial, predictor, atommap);
}

void
PaddleDeepPot::
compute (ENERGYTYPE &			dener,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const int			nghost,
	 const InputNlist &		lmp_list,
	 const int&			ago,
	 const std::vector<VALUETYPE> &	fparam,
	 const std::vector<VALUETYPE> &	aparam_)
{
  std::vector<VALUETYPE> dcoord, dforce, aparam;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  // aparam
  if (daparam > 0){
    aparam.resize(bkw_map.size());
    select_map<VALUETYPE>(aparam, aparam_, fwd_map, daparam);
  }
  // internal nlist
  if (ago == 0){
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);  
  }
  compute_inner(dener, dforce, dvirial, dcoord, datype, dbox, nghost_real, ago, fparam, aparam);
  // bkw map
  dforce_.resize(fwd_map.size() * 3);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3);
}


void
PaddleDeepPot::
compute (ENERGYTYPE &			dener,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial,
	 std::vector<VALUETYPE> &	datom_energy_,
	 std::vector<VALUETYPE> &	datom_virial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox,
	 const std::vector<VALUETYPE> &	fparam,
	 const std::vector<VALUETYPE> &	aparam)
{
  atommap = deepmd::AtomMap<VALUETYPE> (datype_.begin(), datype_.end());
  validate_fparam_aparam(atommap.get_type().size(), fparam, aparam, dfparam, daparam);

  int nloc = paddle_input_tensors (predictor, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam, atommap);

  paddle_run_model (dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor, atommap);
}

void
PaddleDeepPot::
compute (ENERGYTYPE &			dener,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial,
	 std::vector<VALUETYPE> &	datom_energy_,
	 std::vector<VALUETYPE> &	datom_virial_,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const int			nghost, 
	 const InputNlist &	lmp_list,
	 const int               &	ago,
	 const std::vector<VALUETYPE> &	fparam,
	 const std::vector<VALUETYPE> &	aparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam, dfparam, daparam);
  if (ago == 0) {
    atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
    assert (nloc == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }
  int ret = paddle_input_tensors (predictor, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);
  assert (nloc == ret);
  paddle_run_model (dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor, atommap, nghost);
}

void
PaddleDeepPot::
compute_inner (ENERGYTYPE &			dener,
	       std::vector<VALUETYPE> &		dforce_,
	       std::vector<VALUETYPE> &		dvirial,
	       const std::vector<VALUETYPE> &	dcoord_,
	       const std::vector<int> &		datype_,
	       const std::vector<VALUETYPE> &	dbox, 
	       const int			nghost,
	       const int&			ago,
	       const std::vector<VALUETYPE> &	fparam,
	       const std::vector<VALUETYPE> &	aparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;

  validate_fparam_aparam(nloc, fparam, aparam, dfparam, daparam);

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0) {
    atommap = deepmd::AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
    assert (nloc == atommap.get_type().size());
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }
  int ret = paddle_input_tensors (predictor, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);
  assert (nloc == ret);
  paddle_run_model (dener, dforce_, dvirial, predictor, atommap, nghost);
}

PaddleDeepPotModelDevi::PaddleDeepPotModelDevi ()
    : inited (false),
      init_nbor (false),
      numb_models (0)
{

}

PaddleDeepPotModelDevi::PaddleDeepPotModelDevi (std::vector<std::string> & prog_strs, std::vector<std::string> & params_strs)
{
  init(prog_strs, params_strs);
}

PaddleDeepPotModelDevi::~PaddleDeepPotModelDevi() {}

void PaddleDeepPotModelDevi::init (std::vector<std::string>& prog_strs, std::vector<std::string>& params_strs){
  if (inited){
    std::cerr << "Init error" << std::endl;
    return ;
  }
  numb_models = prog_strs.size();
  configs.resize(numb_models);
  predictors.resize(numb_models);
  math_lib_num_threads = deepmd::get_math_lib_num_threads();
  for (unsigned ii = 0; ii < numb_models; ++ii){
    configs[ii].SetModelBuffer(prog_strs[ii].c_str(), prog_strs[ii].size(), params_strs[ii].c_str(), params_strs[ii].size());
    configs[ii].EnableMKLDNN();
    configs[ii].SetCpuMathLibraryNumThreads(math_lib_num_threads);
    configs[ii].SwitchIrOptim();
    configs[ii].EnableMemoryOptim();
    predictors[ii] = paddle_infer::CreatePredictor(configs[ii]);
  }

  /*
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut")
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam_ = get_scalar<int>("fitting_attr/dfparam");
  daparam_ = get_scalar<int>("fitting_attr/daparam");
  dfparam_ = dfparam_ < 0 ? 0 : dfparam_;
  daparam_ = daparam_ < 0 ? 0 : daparam_;
  */
  cell_size = rcut;
  dfparam = 0;
  daparam = 0;
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;

  inited = true;
  init_nbor = false;
}

void  
PaddleDeepPotModelDevi::
cum_sum (const std::vector<std::vector<int> > n_sel) 
{
    for (int ii = 0; ii < numb_models; ++ii) {
        std::vector<int> _sec;
        _sec.resize (n_sel[ii].size() + 1);
        _sec[0] = 0;
        for (int jj = 1; jj < _sec.size(); ++jj) {
            _sec[jj] = _sec[jj-1] + n_sel[ii][jj-1];
        }
        sec.push_back(_sec);
    }
}

void
PaddleDeepPotModelDevi::
validate_fparam_aparam(const int & nloc,
		       const std::vector<VALUETYPE> &fparam,
		       const std::vector<VALUETYPE> &aparam,
           const int &dfparam, 
           const int &daparam) const 
{
  if (fparam.size() != dfparam) {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }
  if (aparam.size() != daparam * nloc) {
    throw std::runtime_error("the dim of atom parameter provided is not consistent with what the model uses");
  }  
}

void
PaddleDeepPotModelDevi::
compute (std::vector<ENERGYTYPE> &		all_energy,
	 std::vector<std::vector<VALUETYPE>> &	all_force,
	 std::vector<std::vector<VALUETYPE>> &	all_virial,
	 const std::vector<VALUETYPE> &		dcoord_,
	 const std::vector<int> &		datype_,
	 const std::vector<VALUETYPE> &		dbox,
	 const int				nghost,
	 const InputNlist &		lmp_list,
	 const int                &		ago,
	 const std::vector<VALUETYPE> &		fparam,
	 const std::vector<VALUETYPE> &		aparam)
{
  if (numb_models == 0) return;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam, dfparam, daparam);

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0) {
    atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
    assert (nloc == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }

  std::vector<int> ret(numb_models, 0);
  for (unsigned ii = 0; ii < numb_models; ++ii){
    ret[ii] = paddle_input_tensors (predictors[ii], dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);
  }
  
  all_energy.resize (numb_models);
  all_force.resize (numb_models);
  all_virial.resize (numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii){
    assert (nloc == ret[ii]);
  }
  for (unsigned ii = 0; ii < numb_models; ++ii) {
      // paddle_run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii], predictor_, atommap, nghost); // we may need to open new context
      paddle_run_model (all_energy[ii], all_force[ii], all_virial[ii], predictors[ii], atommap, nghost);
  }
}

void
PaddleDeepPotModelDevi::
compute (std::vector<ENERGYTYPE> &		all_energy,
	 std::vector<std::vector<VALUETYPE>> &	all_force,
	 std::vector<std::vector<VALUETYPE>> &	all_virial,
	 std::vector<std::vector<VALUETYPE>> &	all_atom_energy,
	 std::vector<std::vector<VALUETYPE>> &	all_atom_virial,
	 const std::vector<VALUETYPE> &		dcoord_,
	 const std::vector<int> &		datype_,
	 const std::vector<VALUETYPE> &		dbox,
	 const int				nghost,
	 const InputNlist &		lmp_list,
	 const int	             &		ago,
	 const std::vector<VALUETYPE> &	 	fparam,
	 const std::vector<VALUETYPE> &	 	aparam)
{
  if (numb_models == 0) return;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam, dfparam, daparam);

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0) {
      atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
      assert (nloc == atommap.get_type().size());

      nlist_data.copy_from_nlist(lmp_list);
      nlist_data.shuffle(atommap);
      nlist_data.make_inlist(nlist);
  }

  std::vector<int> ret(numb_models, 0);
  for (int ii = 0; ii < numb_models; ++ii){
    ret[ii] = paddle_input_tensors (predictors[ii], dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);
  }

  all_energy.resize (numb_models);
  all_force .resize (numb_models);
  all_virial.resize (numb_models);
  all_atom_energy.resize (numb_models);
  all_atom_virial.resize (numb_models); 
  for (unsigned ii = 0; ii < numb_models; ++ii){
    assert (nloc == ret[ii]);
  }
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    paddle_run_model (all_energy[ii], all_force[ii], all_virial[ii], all_atom_energy[ii], all_atom_virial[ii], predictors[ii], atommap, nghost);
  }
}

#ifndef HIGH_PREC
void
PaddleDeepPotModelDevi::
compute_avg (ENERGYTYPE &		dener, 
	     const std::vector<ENERGYTYPE >&	all_energy) 
{
  assert (all_energy.size() == numb_models);
  if (numb_models == 0) return;

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    dener += all_energy[ii];
  }
  dener /= (ENERGYTYPE)(numb_models);  
}
#endif

void
PaddleDeepPotModelDevi::
compute_avg (VALUETYPE &		dener, 
	     const std::vector<VALUETYPE > &	all_energy) 
{
  assert (all_energy.size() == numb_models);
  if (numb_models == 0) return;

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    dener += all_energy[ii];
  }
  dener /= (VALUETYPE)(numb_models);  
}

void
PaddleDeepPotModelDevi::
compute_avg (std::vector<VALUETYPE> &		avg, 
	     const std::vector<std::vector<VALUETYPE> > &	xx) 
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;
  
  avg.resize(xx[0].size());
  fill (avg.begin(), avg.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii){
    for (unsigned jj = 0; jj < avg.size(); ++jj){
      avg[jj] += xx[ii][jj];
    }
  }

  for (unsigned jj = 0; jj < avg.size(); ++jj){
    avg[jj] /= VALUETYPE(numb_models);
  }
}

void
PaddleDeepPotModelDevi::
compute_std_e (std::vector<VALUETYPE> &		std, 
	       const std::vector<VALUETYPE> &	avg, 
	       const std::vector<std::vector<VALUETYPE> >&xx)  
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof;
  assert (nloc == ndof);
  
  std.resize(nloc);
  fill (std.begin(), std.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0 ; jj < nloc; ++jj){
      const VALUETYPE * tmp_f = &(xx[ii][jj]);
      const VALUETYPE * tmp_avg = &(avg[jj]);
      VALUETYPE vdiff = xx[ii][jj] - avg[jj];
      std[jj] += vdiff * vdiff;
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj){
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
    // std[jj] = sqrt(std[jj] / VALUETYPE(numb_models-1));
  }
}

void
PaddleDeepPotModelDevi::
compute_std_f (std::vector<VALUETYPE> &		std, 
	       const std::vector<VALUETYPE> &	avg, 
	       const std::vector<std::vector<VALUETYPE> >&xx)  
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof / 3;
  assert (nloc * 3 == ndof);
  
  std.resize(nloc);
  fill (std.begin(), std.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0 ; jj < nloc; ++jj){
      const VALUETYPE * tmp_f = &(xx[ii][jj*3]);
      const VALUETYPE * tmp_avg = &(avg[jj*3]);
      VALUETYPE vdiff[3];
      vdiff[0] = tmp_f[0] - tmp_avg[0];
      vdiff[1] = tmp_f[1] - tmp_avg[1];
      vdiff[2] = tmp_f[2] - tmp_avg[2];
      std[jj] += deepmd::dot3(vdiff, vdiff);
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj){
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
    // std[jj] = sqrt(std[jj] / VALUETYPE(numb_models-1));
  }
}

void
PaddleDeepPotModelDevi::
compute_relative_std_f (std::vector<VALUETYPE> &std,
			const std::vector<VALUETYPE> &avg,
			const VALUETYPE eps)
{
  unsigned nloc = std.size();
  for (unsigned ii = 0; ii < nloc; ++ii){
    const VALUETYPE * tmp_avg = &(avg[ii*3]);
      VALUETYPE vdiff[3];
      vdiff[0] = tmp_avg[0];
      vdiff[1] = tmp_avg[1];
      vdiff[2] = tmp_avg[2];
      VALUETYPE f_norm = sqrt(deepmd::dot3(vdiff, vdiff));
      // relative std = std/(abs(f)+eps)
      std[ii] /= f_norm + eps;
  }
}


