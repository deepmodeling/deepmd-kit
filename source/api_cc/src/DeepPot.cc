#include "DeepPot.h"
#include "AtomMap.h"
#include <stdexcept>	
#include "device.h"
#include <dlfcn.h>

using namespace tensorflow;
using namespace deepmd;

static 
std::vector<int> cum_sum (const std::vector<int32> & n_sel) {
    std::vector<int> sec;
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii) {
        sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
    return sec;
}

static void 
run_model (ENERGYTYPE &			dener,
	   std::vector<VALUETYPE> &	dforce_,
	   std::vector<VALUETYPE> &	dvirial,
	   Session *			session, 
	   const std::vector<std::pair<std::string, Tensor>> & input_tensors,
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
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(9);
    fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
    return;
  }

  std::vector<Tensor> output_tensors;
  check_status (session->Run(input_tensors, 
			    {"o_energy", "o_force", "o_atom_energy", "o_atom_virial"}, 
			    {}, 
			    &output_tensors));
  
  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_av = output_tensors[3];

  auto oe = output_e.flat <ENERGYTYPE> ();
  auto of = output_f.flat <VALUETYPE> ();
  auto oav = output_av.flat <VALUETYPE> ();

  dener = oe(0);
  std::vector<VALUETYPE> dforce (3 * nall);
  dvirial.resize (9);
  for (unsigned ii = 0; ii < nall * 3; ++ii){
    dforce[ii] = of(ii);
  }
  // set dvirial to zero, prevent input vector is not zero (#1123)
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  for (int ii = 0; ii < nall; ++ii) {
    dvirial[0] += (VALUETYPE)1.0 * oav(9*ii+0);
    dvirial[1] += (VALUETYPE)1.0 * oav(9*ii+1);
    dvirial[2] += (VALUETYPE)1.0 * oav(9*ii+2);
    dvirial[3] += (VALUETYPE)1.0 * oav(9*ii+3);
    dvirial[4] += (VALUETYPE)1.0 * oav(9*ii+4);
    dvirial[5] += (VALUETYPE)1.0 * oav(9*ii+5);
    dvirial[6] += (VALUETYPE)1.0 * oav(9*ii+6);
    dvirial[7] += (VALUETYPE)1.0 * oav(9*ii+7);
    dvirial[8] += (VALUETYPE)1.0 * oav(9*ii+8);
  }
  dforce_ = dforce;
  atommap.backward (dforce_.begin(), dforce.begin(), 3);
}

#if HUAWEI_ASCEND
static void 
run_model (ENERGYTYPE &			dener,
	   std::vector<VALUETYPE> &	dforce_,
	   std::vector<VALUETYPE> &	dvirial,
	   Session *			session, 
	   std::vector<std::pair<std::string, Tensor>> & input_tensors,
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
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(9);
    fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
    return;
  }

  std::vector<Tensor> output_tensors;
  if (input_tensors.size() == 4) {
      std::vector<std::pair<std::string, Tensor>> temp_input_tensors;
      temp_input_tensors.push_back(input_tensors[3]);
      std::vector<Tensor> tmp_output_tensor;
      check_status(session->Run(temp_input_tensors,
                                {},
                                {"Variable_mesh/Assign"},
                                &tmp_output_tensor));
      input_tensors.pop_back();
  }
  check_status (session->Run(input_tensors,
			    {"o_force", "o_atom_energy", "o_atom_virial"},
			    {},
			    &output_tensors));
  Tensor output_f = output_tensors[0];
  Tensor output_ae = output_tensors[1];
  Tensor output_av = output_tensors[2];

  auto of = output_f.flat <VALUETYPE> ();
  auto oae = output_ae.flat <VALUETYPE> ();
  auto oav = output_av.flat <VALUETYPE> ();

  //de-offset coord
  Tensor type = input_tensors[1].second;
  auto type_val = type.flat <int> ();
  int oae_idx = 0;
  double oae_sum = 0.0;
  std::vector<VALUETYPE> dforce (3 * nall);
  dvirial.resize (9);
  // set dvirial to zero, prevent input vector is not zero (#1123)
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  for (int ii = 0; ii < type_val.size(); ++ii) {
      if (type_val(ii) != -1) {
          if (oae_idx < nloc) {
              oae_sum += static_cast<double>(oae(ii));
          }
          dforce[oae_idx*3] = of(ii*3);
          dforce[oae_idx*3 + 1] = of(ii*3 + 1);
          dforce[oae_idx*3 + 2] = of(ii*3 + 2);
          dvirial[0] += 1.0 * oav(9*ii+0);
          dvirial[1] += 1.0 * oav(9*ii+1);
          dvirial[2] += 1.0 * oav(9*ii+2);
          dvirial[3] += 1.0 * oav(9*ii+3);
          dvirial[4] += 1.0 * oav(9*ii+4);
          dvirial[5] += 1.0 * oav(9*ii+5);
          dvirial[6] += 1.0 * oav(9*ii+6);
          dvirial[7] += 1.0 * oav(9*ii+7);
          dvirial[8] += 1.0 * oav(9*ii+8);
          oae_idx += 1;
      }
  }
  dener = oae_sum;
  dforce_ = dforce;
  atommap.backward (dforce_.begin(), dforce.begin(), 3);
}
#endif

static void run_model (ENERGYTYPE   &		dener,
		       std::vector<VALUETYPE>&	dforce_,
		       std::vector<VALUETYPE>&	dvirial,	   
		       std::vector<VALUETYPE>&	datom_energy_,
		       std::vector<VALUETYPE>&	datom_virial_,
		       Session*			session, 
		       const std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const deepmd::AtomMap<VALUETYPE> &   atommap, 
		       const int&		nghost = 0)
{
    unsigned nloc = atommap.get_type().size();
    unsigned nall = nloc + nghost;
    if (nloc == 0) {
        dener = 0;
        // no backward map needed
        // dforce of size nall * 3
        dforce_.resize(nall * 3);
        fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
        // dvirial of size 9
        dvirial.resize(9);
        fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
        // datom_energy_ of size nall
        datom_energy_.resize(nall);
        fill(datom_energy_.begin(), datom_energy_.end(), (VALUETYPE)0.0);
        // datom_virial_ of size nall * 9
        datom_virial_.resize(nall * 9);
        fill(datom_virial_.begin(), datom_virial_.end(), (VALUETYPE)0.0);
        return;
    }
    std::vector<Tensor> output_tensors;

    check_status (session->Run(input_tensors, 
			    {"o_energy", "o_force", "o_atom_energy", "o_atom_virial"}, 
			    {},
			    &output_tensors));

    Tensor output_e = output_tensors[0];
    Tensor output_f = output_tensors[1];
    Tensor output_ae = output_tensors[2];
    Tensor output_av = output_tensors[3];

    auto oe = output_e.flat <ENERGYTYPE> ();
    auto of = output_f.flat <VALUETYPE> ();
    auto oae = output_ae.flat <VALUETYPE> ();
    auto oav = output_av.flat <VALUETYPE> ();

    dener = oe(0);
    std::vector<VALUETYPE> dforce (3 * nall);
    std::vector<VALUETYPE> datom_energy (nall, 0);
    std::vector<VALUETYPE> datom_virial (9 * nall);
    dvirial.resize (9);
    for (int ii = 0; ii < nall * 3; ++ii) {
        dforce[ii] = of(ii);
    }
    for (int ii = 0; ii < nloc; ++ii) {
        datom_energy[ii] = oae(ii);
    }
    for (int ii = 0; ii < nall * 9; ++ii) {
        datom_virial[ii] = oav(ii);
    }
    // set dvirial to zero, prevent input vector is not zero (#1123)
    std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
    for (int ii = 0; ii < nall; ++ii) {
        dvirial[0] += (VALUETYPE)1.0 * datom_virial[9*ii+0];
        dvirial[1] += (VALUETYPE)1.0 * datom_virial[9*ii+1];
        dvirial[2] += (VALUETYPE)1.0 * datom_virial[9*ii+2];
        dvirial[3] += (VALUETYPE)1.0 * datom_virial[9*ii+3];
        dvirial[4] += (VALUETYPE)1.0 * datom_virial[9*ii+4];
        dvirial[5] += (VALUETYPE)1.0 * datom_virial[9*ii+5];
        dvirial[6] += (VALUETYPE)1.0 * datom_virial[9*ii+6];
        dvirial[7] += (VALUETYPE)1.0 * datom_virial[9*ii+7];
        dvirial[8] += (VALUETYPE)1.0 * datom_virial[9*ii+8];
	}
    dforce_ = dforce;
    datom_energy_ = datom_energy;
    datom_virial_ = datom_virial;
    atommap.backward (dforce_.begin(), dforce.begin(), 3);
    atommap.backward (datom_energy_.begin(), datom_energy.begin(), 1);
    atommap.backward (datom_virial_.begin(), datom_virial.begin(), 9);
}


DeepPot::
DeepPot ()
    : inited (false), init_nbor (false),
      graph_def(new GraphDef())
{
}

DeepPot::
DeepPot (const std::string & model, const int & gpu_rank, const std::string & file_content)
    : inited (false), init_nbor (false),
      graph_def(new GraphDef())
{
  init(model, gpu_rank, file_content);  
}

DeepPot::~DeepPot() {
  delete graph_def;
  delete session;
}

void
DeepPot::
init (const std::string & model, const int & gpu_rank, const std::string & file_content)
{
  if (inited){
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
    return ;
  }
  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  deepmd::load_op_library();

  if(file_content.size() == 0)
    check_status (ReadBinaryProto(Env::Default(), model, graph_def));
  else
    (*graph_def).ParseFromString(file_content);
  int gpu_num = -1;
  #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  DPGetDeviceCount(gpu_num); // check current device environment
  if (gpu_num > 0) {
    options.config.set_allow_soft_placement(true);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
    std::string str = "/gpu:";
    str += std::to_string(gpu_rank % gpu_num);
    graph::SetDefaultDevice(str, graph_def);
  }
  #endif // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  check_status (NewSession(options, &session));
  check_status (session->Create(*graph_def));
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  try{
  model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  } catch (deepmd::tf_exception& e){
    // no model version defined in old models
    model_version = "0.0";
  }
  if(! model_compatable(model_version)){
    throw deepmd::deepmd_exception(
	"incompatable model: version " + model_version 
	+ " in graph, but version " + global_model_version 
	+ " supported ");
  }
  inited = true;
  
  init_nbor = false;
}

#if HUAWEI_ASCEND
void
DeepPot::
init (const int & npu_rank)
{
  if (inited){
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
    return ;
  }
  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  deepmd::load_op_library();

  // init NPU session 
  auto* custom_op = options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_op->set_name("NpuOptimizer");
  auto* params = custom_op->mutable_parameter_map();
  ::tensorflow::AttrValue use_off_line;
  ::tensorflow::AttrValue mix_compile_mode;
  ::tensorflow::AttrValue op_debug_level;
  ::tensorflow::AttrValue precision_mode;
  use_off_line.set_b(true);
  mix_compile_mode.set_b(false);
  op_debug_level.set_i(0);
  precision_mode.set_s("must_keep_origin_dtype");
  params->insert({"use_off_line", use_off_line});
  params->insert({"mix_compile_mode", mix_compile_mode});
  params->insert({"op_debug_level", op_debug_level});
  params->insert({"precision_mode", precision_mode});

  // set visible ascend devices
  char *pathvar;
  pathvar = getenv("ASCEND_VISIBLE_DEVICES");
  std::string space_delimiter = ",";
  std::vector<string> words{};
  if (pathvar && *pathvar != '\0') {
    std::string pathvar_str = pathvar;
    size_t pos = 0;
    while ((pos = pathvar_str.find(space_delimiter)) != string::npos) {
        words.push_back(pathvar_str.substr(0, pos));
        pathvar_str = pathvar_str.substr(pos + space_delimiter.length());
    }
    words.push_back(pathvar_str.substr(0, pathvar_str.size()));
    ::tensorflow::AttrValue session_device_id;
    int dev_id = std::stoi(words[npu_rank % words.size()]);
    session_device_id.set_i(dev_id);
    params->insert({"session_device_id", session_device_id});
  } else {
    std::cout << "CAN NOT GET ASCEND_VISIBLE_DEVICES!" << std::endl;
    ::tensorflow::AttrValue session_device_id;
    session_device_id.set_i(npu_rank);
    params->insert({"session_device_id", session_device_id});
  }
  options.config.mutable_graph_options()->mutable_rewrite_options()->set_remapping(::tensorflow::RewriterConfig_Toggle::RewriterConfig_Toggle_OFF);
  check_status (NewSession(options, &session));
  check_status (session->Create(*graph_def));
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  try{
  model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  } catch (deepmd::tf_exception& e){
    // no model version defined in old models
    model_version = "0.0";
  }
  if(! model_compatable(model_version)){
    throw deepmd::deepmd_exception(
	"incompatable model: version " + model_version 
	+ " in graph, but version " + global_model_version 
	+ " supported ");
  }
  inited = true;
  
  init_nbor = false;
}

void
DeepPot::
init_graph (const std::string & model, const std::vector<int > & type_count, const std::string & file_content)
{
  
  void *handler = dlopen("_tf_adapter.so", RTLD_NOW | RTLD_GLOBAL);
  if (handler == nullptr) {
      printf("%s\n", dlerror());
      std::cout << "_tf_adapter.so open failed!" << std::endl;
      return;
  }
  if(file_content.size() == 0){
    check_status (ReadBinaryProto(Env::Default(), model, graph_def));
  }
  else
    (*graph_def).ParseFromString(file_content);
  int nloc_padding = 0;
  int nall_padding;
  char *nallVar;
  nallVar = getenv("NALLVAL");
  if (nallVar && *nallVar != '\0') {
    std::string nall_string = nallVar;
    nall_padding = std::stoi(nall_string);
  } else {
    std::cout << "CAN NOT GET NALLVAL! USE DEFAULT VALUE: 30000" << std::endl;
    nall_padding = 30000;
  }
  natoms_padding = {nloc_padding, nall_padding};
  for (int i = 0; i < type_count.size(); i++) {
    int type_i_padding = type_count[i] * 1.1;
    natoms_padding.push_back(type_i_padding);
    nloc_padding += type_i_padding;
  }
  natoms_padding[0] = nloc_padding;
  int natoms_padding_dim = natoms_padding.size();
  for (uint32_t i = 0; i < (*graph_def).node_size(); i++) {
      // transfer the natoms placeholder op to a const op
      auto* node = (*graph_def).mutable_node(i);
      if (node->name() == "t_natoms") {
          node->set_op("Const");
          auto* natoms_attr = node->mutable_attr();
          ::tensorflow::AttrValue natoms_V;
          auto *natoms_tensor = natoms_V.mutable_tensor();
          natoms_tensor->set_dtype(DataType::DT_INT32);
          ::tensorflow::TensorShapeProto tensorShapeProto;
          ::tensorflow::TensorShapeProto_Dim* dim_0 = tensorShapeProto.add_dim();
          dim_0->set_size(natoms_padding_dim);
          *natoms_tensor->mutable_tensor_shape() = tensorShapeProto;
          port::CopyFromArray(natoms_tensor->mutable_tensor_content(), reinterpret_cast<const char*> \
          (natoms_padding.data()), natoms_padding_dim*sizeof(int));
          natoms_attr->insert({"value", natoms_V});
      }
      
      // transfer the mesh placeholder op to a assign op
      if (node->name() == "t_mesh") {
          // add Variablve node
          auto *mesh_var = (*graph_def).add_node();
          mesh_var->set_op("VariableV2");
          mesh_var->set_name("Variable_mesh");
          auto* mesh_var_attr = mesh_var->mutable_attr();
          ::tensorflow::AttrValue container;
          ::tensorflow::AttrValue shared_name;
          ::tensorflow::AttrValue var_dtype;
          container.set_s("");
          shared_name.set_s("");
          var_dtype.set_type(DataType::DT_INT32);
          mesh_var_attr->insert({"container", container});
          mesh_var_attr->insert({"shared_name", shared_name});
          mesh_var_attr->insert({"dtype", var_dtype});
          mesh_var_attr->insert({"shape", node->attr().at("shape")});
          mesh_var_attr->at("shape").mutable_shape()->mutable_dim(0)->set_size(natoms_padding[0]*1026+1); 

          // add Variable Assign node
          auto *mesh_var_assgin = (*graph_def).add_node();
          mesh_var_assgin->set_op("Assign");
          mesh_var_assgin->set_name("Variable_mesh/Assign");
          auto* mesh_var_assgin_attr = mesh_var_assgin->mutable_attr();
          ::tensorflow::AttrValue assign_T;
          ::tensorflow::AttrValue assign_class;
          ::tensorflow::AttrValue assign_locking;
          ::tensorflow::AttrValue assign_validate;
          assign_T.set_type(DataType::DT_INT32);
          auto *assign_lv = assign_class.mutable_list();
          assign_lv->add_s("loc:@Variable");
          assign_locking.set_b(true);
          assign_validate.set_b(false);
          mesh_var_assgin_attr->insert({"T", assign_T});
          mesh_var_assgin_attr->insert({"_class", assign_class});
          mesh_var_assgin_attr->insert({"use_locking", assign_locking});
          mesh_var_assgin_attr->insert({"validate_shape", assign_validate});
          mesh_var_assgin->add_input("Variable_mesh");
          mesh_var_assgin->add_input("t_mesh");

          // add Variable Read node
          auto *mesh_var_read = (*graph_def).add_node();
          mesh_var_read->set_op("Identity");
          mesh_var_read->set_name("Variable_mesh/Read");
          auto* mesh_var_read_attr = mesh_var_read->mutable_attr();
          ::tensorflow::AttrValue read_T;
          ::tensorflow::AttrValue read_class;
          read_T.set_type(DataType::DT_INT32);
          auto *read_lv = read_class.mutable_list();
          read_lv->add_s("loc:@Variable");
          mesh_var_read_attr->insert({"T", read_T});
          mesh_var_read_attr->insert({"_class", read_class});
          mesh_var_read->add_input("Variable_mesh");
      }
      // modify ProdEnvMatA input
      if (node->name() == "ProdEnvMatA") {
          node->set_input(4, "Variable_mesh/Read");
      }
  }
}
#endif //HUAWEI_ASCEND

void 
DeepPot::
print_summary(const std::string &pre) const
{
  std::cout << pre << "installed to:       " + global_install_prefix << std::endl;
  std::cout << pre << "source:             " + global_git_summ << std::endl;
  std::cout << pre << "source branch:       " + global_git_branch << std::endl;
  std::cout << pre << "source commit:      " + global_git_hash << std::endl;
  std::cout << pre << "source commit at:   " + global_git_date << std::endl;
  std::cout << pre << "surpport model ver.:" + global_model_version << std::endl;
  std::cout << pre << "build float prec:   " + global_float_prec << std::endl;
#if defined(GOOGLE_CUDA)
  std::cout << pre << "build variant:      cuda" << std::endl;
#elif defined(TENSORFLOW_USE_ROCM)
  std::cout << pre << "build variant:      rocm" << std::endl;
#else
  std::cout << pre << "build variant:      cpu" << std::endl;
#endif
  std::cout << pre << "build with tf inc:  " + global_tf_include_dir << std::endl;
  std::cout << pre << "build with tf lib:  " + global_tf_lib << std::endl;
  std::cout << pre << "set tf intra_op_parallelism_threads: " <<  num_intra_nthreads << std::endl;
  std::cout << pre << "set tf inter_op_parallelism_threads: " <<  num_inter_nthreads << std::endl;
}

template<class VT>
VT
DeepPot::
get_scalar (const std::string & name) const
{
  return session_get_scalar<VT>(session, name);
}

std::string graph_info(const GraphDef & graph_def) {
    // std::stringstream buffer;
    // std::streambuf * old = std::cout.rdbuf(buffer.rdbuf());
    std::string str = "";
    for (int ii = 0; ii < graph_def.node_size(); ii++) {
        if (graph_def.node(ii).name() == "DescrptSeA") {
            // str = graph_def.node(ii).PrintDebugString();
            str = graph_def.node(ii).DebugString();
            return str;
            // std::cout << str << std::endl;
        }
        if (graph_def.node(ii).name() == "DescrptSeR") {
            // str = graph_def.node(ii).PrintDebugString();
            str = graph_def.node(ii).DebugString();
            return str;
            // std::cout << str << std::endl;
        }
    }
    return str;
}

// init the tmp array data
std::vector<int> DeepPot::get_sel_a () const {
    std::vector<int> sel_a;
    std::istringstream is(graph_info(*graph_def));
    std::string line = "";
    while(is >> line) {
        if (line.find("sel_a") != line.npos) {
            while (std::getline(is, line) && line != "}") {
                if (line.find("i:") != line.npos) {
                    sel_a.push_back(atoi((line.substr(line.find("i:") + 2)).c_str()));
                }
            } break;
        }
        if (line.find("sel") != line.npos) {
            while (std::getline(is, line) && line != "}") {
                if (line.find("i:") != line.npos) {
                    sel_a.push_back(atoi((line.substr(line.find("i:") + 2)).c_str()));
                }
            } break;
        }
    }
    return sel_a;
}

void
DeepPot::
validate_fparam_aparam(const int & nloc,
		       const std::vector<VALUETYPE> &fparam,
		       const std::vector<VALUETYPE> &aparam)const 
{
  if (fparam.size() != dfparam) {
    throw deepmd::deepmd_exception("the dim of frame parameter provided is not consistent with what the model uses");
  }
  if (aparam.size() != daparam * nloc) {
    throw deepmd::deepmd_exception("the dim of atom parameter provided is not consistent with what the model uses");
  }  
}

void
DeepPot::
compute (ENERGYTYPE &			dener,
	 std::vector<VALUETYPE> &	dforce_,
	 std::vector<VALUETYPE> &	dvirial,
	 const std::vector<VALUETYPE> &	dcoord_,
	 const std::vector<int> &	datype_,
	 const std::vector<VALUETYPE> &	dbox, 
	 const std::vector<VALUETYPE> &	fparam,
	 const std::vector<VALUETYPE> &	aparam)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  atommap = deepmd::AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
  assert (nloc == atommap.get_type().size());
  validate_fparam_aparam(nloc, fparam, aparam);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam, atommap);
  assert (ret == nloc);

  run_model (dener, dforce_, dvirial, session, input_tensors, atommap);
}

void
DeepPot::
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
    aparam.resize(bkw_map.size() - nghost_real);
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
DeepPot::
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

    validate_fparam_aparam(nloc, fparam, aparam);
    std::vector<std::pair<std::string, Tensor>> input_tensors;

    // agp == 0 means that the LAMMPS nbor list has been updated
    if (ago == 0) {
      atommap = deepmd::AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
      assert (nloc == atommap.get_type().size());
      nlist_data.shuffle(atommap);
      nlist_data.make_inlist(nlist);
    }
    #if HUAWEI_ASCEND
    int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago, natoms_padding);
    #else
    int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);
    #endif //HUAWEI_ASCEND
    assert (nloc == ret);
    run_model (dener, dforce_, dvirial, session, input_tensors, atommap, nghost);
}


void
DeepPot::
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
  validate_fparam_aparam(atommap.get_type().size(), fparam, aparam);

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int nloc = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam, atommap);

  run_model (dener, dforce_, dvirial, datom_energy_, datom_virial_, session, input_tensors, atommap);
}



void
DeepPot::
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
	 const std::vector<VALUETYPE> &	aparam_)
{
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam_);
    std::vector<std::pair<std::string, Tensor>> input_tensors;
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  int nall_real = bkw_map.size();
  int nloc_real = nall_real - nghost_real;
  dcoord.resize(nall_real * 3);
  datype.resize(nall_real);
  datom_energy.resize(nall_real);
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  select_map<VALUETYPE>(datom_energy, datom_energy_, fwd_map, 1);
  // aparam
  if (daparam > 0){
    aparam.resize(nloc_real);
    select_map<VALUETYPE>(aparam, aparam_, fwd_map, daparam);
  }
    if (ago == 0) {
    atommap = AtomMap<VALUETYPE> (datype.begin(), datype.begin() + nloc_real);
    assert (nloc_real == atommap.get_type().size());

        nlist_data.copy_from_nlist(lmp_list);
        nlist_data.shuffle(atommap);
	nlist_data.make_inlist(nlist);
    }

  int ret = session_input_tensors (input_tensors, dcoord, ntypes, datype, dbox, nlist, fparam, aparam, atommap, nghost_real, ago);
  assert (nloc_real == ret);
  run_model (dener, dforce, dvirial, datom_energy, datom_virial, session, input_tensors, atommap, nghost_real);

  // bkw map
  dforce_.resize(fwd_map.size() * 3);
  datom_energy_.resize(fwd_map.size());
  datom_virial_.resize(fwd_map.size() * 9);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3);
  select_map<VALUETYPE>(datom_energy_, datom_energy, bkw_map, 1);
  select_map<VALUETYPE>(datom_virial_, datom_virial, bkw_map, 9);
}

void
DeepPot::
get_type_map(std::string & type_map){
    type_map = get_scalar<STRINGTYPE>("model_attr/tmap");
}



DeepPotModelDevi::
DeepPotModelDevi ()
    : inited (false), 
      init_nbor (false),
      numb_models (0)
{
}

DeepPotModelDevi::
DeepPotModelDevi (const std::vector<std::string> & models, const int & gpu_rank, const std::vector<std::string> & file_contents)
    : inited (false), 
      init_nbor(false),
      numb_models (0)
{
  init(models, gpu_rank, file_contents);
}

DeepPotModelDevi::~DeepPotModelDevi() {
  for (unsigned ii = 0; ii < numb_models; ++ii){
    delete graph_defs[ii];
  }
}

void
DeepPotModelDevi::
init (const std::vector<std::string> & models, const int & gpu_rank, const std::vector<std::string> & file_contents)
{
  if (inited){
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do nothing at the second call of initializer" << std::endl;
    return ;
  }
  numb_models = models.size();
  sessions.resize(numb_models);
  graph_defs.resize(numb_models);
  
  int gpu_num = -1;
  #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  DPGetDeviceCount(gpu_num);
  #endif // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  for (unsigned ii = 0; ii < numb_models; ++ii){
    graph_defs[ii] = new GraphDef();
    if (file_contents.size() == 0)
      check_status (ReadBinaryProto(Env::Default(), models[ii], graph_defs[ii]));
    else
      (*graph_defs[ii]).ParseFromString(file_contents[ii]);
  }
  #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (gpu_num > 0) {
      options.config.set_allow_soft_placement(true);
      options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
      options.config.mutable_gpu_options()->set_allow_growth(true);
      DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
  }
  #endif // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    if (gpu_num > 0) {
      std::string str = "/gpu:";
      str += std::to_string(gpu_rank % gpu_num);
      graph::SetDefaultDevice(str, &(*graph_defs[ii]));
    }
    check_status (NewSession(options, &(sessions[ii])));
    check_status (sessions[ii]->Create(*graph_defs[ii]));
  }
  rcut = get_scalar<VALUETYPE>("descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  if(! model_compatable(model_version)){
    throw deepmd::deepmd_exception(
	"incompatable model: version " + model_version 
	+ " in graph, but version " + global_model_version 
	+ " supported ");
  }
  // rcut = get_rcut();
  // cell_size = rcut;
  // ntypes = get_ntypes();
  inited = true;
  
  init_nbor = false;
}

template<class VT>
VT
DeepPotModelDevi::
get_scalar(const std::string name) const 
{
  VT myrcut;
  for (unsigned ii = 0; ii < numb_models; ++ii){
    VT ret = session_get_scalar<VT>(sessions[ii], name);
    if (ii == 0){
      myrcut = ret;
    }
    else {
      assert (myrcut == ret);
    }
  }
  return myrcut;
}

// init the tmp array data
std::vector<std::vector<int> > 
DeepPotModelDevi::
get_sel () const 
{
    std::vector<std::vector<int> > sec;
    for (int ii = 0; ii < numb_models; ii++) {
        std::vector<int> sel;
        std::istringstream is(graph_info(*graph_defs[ii]));
        std::string line = "";
        while(is >> line) {
            if (line.find("sel") != line.npos) {
                while (std::getline(is, line) && line != "}") {
                    if (line.find("i:") != line.npos) {
                        sel.push_back(atoi((line.substr(line.find("i:") + 2)).c_str()));
                    }
                } break;
            }
            if (line.find("sel_a") != line.npos) {
                while (std::getline(is, line) && line != "}") {
                    if (line.find("i:") != line.npos) {
                        sel.push_back(atoi((line.substr(line.find("i:") + 2)).c_str()));
                    }
                } break;
            }
        }
        sec.push_back(sel);
    }
    return sec;
}


void
DeepPotModelDevi::
validate_fparam_aparam(const int & nloc,
		       const std::vector<VALUETYPE> &fparam,
		       const std::vector<VALUETYPE> &aparam)const 
{
  if (fparam.size() != dfparam) {
    throw deepmd::deepmd_exception("the dim of frame parameter provided is not consistent with what the model uses");
  }
  if (aparam.size() != daparam * nloc) {
    throw deepmd::deepmd_exception("the dim of atom parameter provided is not consistent with what the model uses");
  }  
}

// void
// DeepPotModelDevi::
// compute (ENERGYTYPE &			dener,
// 	 std::vector<VALUETYPE> &	dforce_,
// 	 std::vector<VALUETYPE> &	dvirial,
// 	 std::vector<VALUETYPE> &	model_devi,
// 	 const std::vector<VALUETYPE> &	dcoord_,
// 	 const std::vector<int> &	datype_,
// 	 const std::vector<VALUETYPE> &	dbox,
// 	 const std::vector<VALUETYPE> &	fparam,
// 	 const std::vector<VALUETYPE> &	aparam)
// {
//   if (numb_models == 0) return;

//   atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.end());
//   validate_fparam_aparam(atommap.get_type().size(), fparam, aparam);

//   std::vector<std::pair<std::string, Tensor>> input_tensors;
//   int nloc = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam, atommap);

//   std::vector<ENERGYTYPE > all_energy (numb_models);
//   std::vector<std::vector<VALUETYPE > > all_force (numb_models);
//   std::vector<std::vector<VALUETYPE > > all_virial (numb_models);

//   for (unsigned ii = 0; ii < numb_models; ++ii){
//     run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii], input_tensors, atommap);
//   }

//   dener = 0;
//   for (unsigned ii = 0; ii < numb_models; ++ii){
//     dener += all_energy[ii];
//   }
//   dener /= VALUETYPE(numb_models);
//   compute_avg (dvirial, all_virial);  
//   compute_avg (dforce_, all_force);
  
//   compute_std_f (model_devi, dforce_, all_force);
  
//   // for (unsigned ii = 0; ii < numb_models; ++ii){
//   //   cout << all_force[ii][573] << " " << all_force[ii][574] << " " << all_force[ii][575] << endl;
//   // }
//   // cout << dforce_[573] << " " 
//   //      << dforce_[574] << " " 
//   //      << dforce_[575] << " " 
//   //      << model_devi[191] << endl;
// }

void
DeepPotModelDevi::
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
  validate_fparam_aparam(nloc, fparam, aparam);
  std::vector<std::pair<std::string, Tensor>> input_tensors;

    // agp == 0 means that the LAMMPS nbor list has been updated
    if (ago == 0) {
        atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
        assert (nloc == atommap.get_type().size());

        nlist_data.copy_from_nlist(lmp_list);
        nlist_data.shuffle(atommap);
	nlist_data.make_inlist(nlist);
    }
    int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);

    all_energy.resize (numb_models);
    all_force.resize (numb_models);
    all_virial.resize (numb_models);
    assert (nloc == ret);
    for (unsigned ii = 0; ii < numb_models; ++ii) {
        run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii], input_tensors, atommap, nghost);
    }
}

void
DeepPotModelDevi::
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
  validate_fparam_aparam(nloc, fparam, aparam);
  std::vector<std::pair<std::string, Tensor>> input_tensors;

    // agp == 0 means that the LAMMPS nbor list has been updated
    if (ago == 0) {
        atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.begin() + nloc);
        assert (nloc == atommap.get_type().size());

        nlist_data.copy_from_nlist(lmp_list);
        nlist_data.shuffle(atommap);
	nlist_data.make_inlist(nlist);
    }
    int ret = session_input_tensors (input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, atommap, nghost, ago);

    all_energy.resize (numb_models);
    all_force .resize (numb_models);
    all_virial.resize (numb_models);
    all_atom_energy.resize (numb_models);
    all_atom_virial.resize (numb_models); 
    assert (nloc == ret);
    for (unsigned ii = 0; ii < numb_models; ++ii) {
        run_model (all_energy[ii], all_force[ii], all_virial[ii], all_atom_energy[ii], all_atom_virial[ii], sessions[ii], input_tensors, atommap, nghost);
    }
}

void
DeepPotModelDevi::
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

#ifndef HIGH_PREC
void
DeepPotModelDevi::
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
DeepPotModelDevi::
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
DeepPotModelDevi::
compute_std (
    std::vector<VALUETYPE> &		std, 
    const std::vector<VALUETYPE> &	avg, 
    const std::vector<std::vector<VALUETYPE> >&xx,
    const int & stride)
{
  assert (xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof / stride;
  assert (nloc * stride == ndof);
  
  std.resize(nloc);
  fill (std.begin(), std.end(), VALUETYPE(0.));
  
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0 ; jj < nloc; ++jj){
      const VALUETYPE * tmp_f = &(xx[ii][jj*stride]);
      const VALUETYPE * tmp_avg = &(avg[jj*stride]);
      for (unsigned dd = 0; dd < stride; ++dd){
	VALUETYPE vdiff = tmp_f[dd] - tmp_avg[dd];
	std[jj] += vdiff * vdiff;
      }
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj){
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
  }
}


void
DeepPotModelDevi::
compute_std_e (std::vector<VALUETYPE> &		std, 
	       const std::vector<VALUETYPE> &	avg, 
	       const std::vector<std::vector<VALUETYPE> >&xx)  
{
  compute_std(std, avg, xx, 1);
}

void
DeepPotModelDevi::
compute_std_f (std::vector<VALUETYPE> &		std, 
	       const std::vector<VALUETYPE> &	avg, 
	       const std::vector<std::vector<VALUETYPE> >&xx)  
{
  compute_std(std, avg, xx, 3);
}

void
DeepPotModelDevi::
compute_relative_std (
    std::vector<VALUETYPE> &std,
    const std::vector<VALUETYPE> &avg,
    const VALUETYPE eps, 
    const int & stride)
{
  unsigned ndof = avg.size();
  unsigned nloc = std.size();
  assert (nloc * stride == ndof);

  for (unsigned ii = 0; ii < nloc; ++ii){
    const VALUETYPE * tmp_avg = &(avg[ii*stride]);
    VALUETYPE f_norm = 0.0;
    for (unsigned dd = 0; dd < stride; ++dd){
      f_norm += tmp_avg[dd] * tmp_avg[dd];
    }
    f_norm = sqrt(f_norm);
    std[ii] /= f_norm + eps;
  }
}

void
DeepPotModelDevi::
compute_relative_std_f (std::vector<VALUETYPE> &std,
			const std::vector<VALUETYPE> &avg,
			const VALUETYPE eps)
{
  compute_relative_std(std, avg, eps, 3);
}

