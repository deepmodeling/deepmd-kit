#include "DeepPotPT.h"

using namespace deepmd;
DeepPotPT::DeepPotPT()
: inited(false) { }
DeepPotPT::DeepPotPT(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false){
  try {
    init(model, gpu_rank);
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}
void DeepPotPT::init(const std::string& model, const int& gpu_rank) {
    if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
    }
    std::cout << "load model from: " <<model <<" to gpu "<< gpu_rank << std::endl;
    gpu_id = gpu_rank;
    torch::Device device(torch::kCUDA, gpu_rank);

    module = torch::jit::load(model,device);                                                                                                       
    torch::jit::FusionStrategy strategy;
    strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
    torch::jit::setFusionStrategy(strategy);

    // at::globalContext().setAllowTF32CuBLAS(true);
    // at::globalContext().setAllowTF32CuDNN(true);
    auto rcut_ = module.run_method("get_rcut").toDouble();
    rcut = static_cast<VALUETYPE>(rcut_);
    inited = true;
}
DeepPotPT::~DeepPotPT() { }



template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
            std::vector<VALUETYPE>& force,
            std::vector<VALUETYPE>& virial,
            const std::vector<VALUETYPE>& coord,
            const std::vector<int>& atype,
            const std::vector<VALUETYPE>& box,
            const InputNlist& lmp_list,
            const int& ago)
{
    torch::Device device(torch::kCUDA, gpu_id);
    std::vector<VALUETYPE> coord_wrapped = coord;
    int natoms = atype.size();
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto int_options = torch::TensorOptions().dtype(torch::kInt64);
    auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
    std::vector<torch::jit::IValue> inputs;
    at::Tensor coord_wrapped_Tensor = torch::from_blob(coord_wrapped.data(), {1,natoms, 3}, options).to(device);
    inputs.push_back(coord_wrapped_Tensor);
    std::vector<int64_t> atype_64(atype.begin(), atype.end());
    at::Tensor atype_Tensor = torch::from_blob(atype_64.data(), {1,natoms}, int_options).to(device);
    inputs.push_back(atype_Tensor);
    if(ago == 0)
    {
      int64_t nnei = module.run_method("get_nnei").toInt();
      nlist_data.copy_from_nlist(lmp_list,max_num_neighbors,nnei);
      if(max_num_neighbors > nnei)
      {
        at::Tensor firstneigh = torch::from_blob(nlist_data.jlist, {lmp_list.inum,max_num_neighbors}, int32_options);
        at::Tensor nlist= firstneigh.to(torch::kInt64).to(device);
        firstneigh_tensor  = module.run_method("sort_neighbor_list",coord_wrapped_Tensor,nlist).toTensor();
      }
      else
      {
        at::Tensor firstneigh = torch::from_blob(nlist_data.jlist, {1,lmp_list.inum,max_num_neighbors}, int32_options);
        firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
      }
    }
    inputs.push_back(firstneigh_tensor);
    c10::Dict<c10::IValue, c10::IValue> outputs = module.forward(inputs).toGenericDict();
    c10::IValue energy_ = outputs.at("energy");
    c10::IValue force_ = outputs.at("extended_force");
    c10::IValue virial_ = outputs.at("extended_virial");
    ener = energy_.toTensor().item<double>();

    torch::Tensor flat_force_ = force_.toTensor().view({-1});
    torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
    force.assign(cpu_force_.data_ptr<double>(), cpu_force_.data_ptr<double>() + cpu_force_.numel());

    torch::Tensor flat_virial_ = virial_.toTensor().view({-1});
    torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
    virial.assign(cpu_virial_.data_ptr<double>(), cpu_virial_.data_ptr<double>() + cpu_virial_.numel());

}
template void DeepPotPT::compute<double, double>(double& ener,
            std::vector<double>& force,
            std::vector<double>& virial,
            const std::vector<double>& coord,
            const std::vector<int>& atype,
            const std::vector<double>& box,
            const InputNlist& lmp_list,
            const int& ago);


template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
            std::vector<VALUETYPE>& force,
            std::vector<VALUETYPE>& virial,
            const std::vector<VALUETYPE>& coord,
            const std::vector<int>& atype,
            const std::vector<VALUETYPE>& box)
{
    auto device = torch::kCUDA;
    module.to(device);
    std::vector<VALUETYPE> coord_wrapped = coord;
    int natoms = atype.size();
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto int_options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<torch::jit::IValue> inputs;
    at::Tensor coord_wrapped_Tensor = torch::from_blob(coord_wrapped.data(), {1, natoms, 3}, options).to(device);
    inputs.push_back(coord_wrapped_Tensor);
    std::vector<int64_t> atype_64(atype.begin(), atype.end());
    at::Tensor atype_Tensor = torch::from_blob(atype_64.data(), {1, natoms}, int_options).to(device);
    inputs.push_back(atype_Tensor);
    at::Tensor box_Tensor = torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options).to(device);
    inputs.push_back(box_Tensor);
    c10::Dict<c10::IValue, c10::IValue> outputs = module.forward(inputs).toGenericDict();


    c10::IValue energy_ = outputs.at("energy");
    c10::IValue force_ = outputs.at("force");
    c10::IValue virial_ = outputs.at("virial");
    ener = energy_.toTensor().item<double>();

    torch::Tensor flat_force_ = force_.toTensor().view({-1});
    torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
    force.assign(cpu_force_.data_ptr<double>(), cpu_force_.data_ptr<double>() + cpu_force_.numel());

    torch::Tensor flat_virial_ = virial_.toTensor().view({-1});
    torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
    virial.assign(cpu_virial_.data_ptr<double>(), cpu_virial_.data_ptr<double>() + cpu_virial_.numel());

}
template void DeepPotPT::compute<double, double>(double& ener,
            std::vector<double>& force,
            std::vector<double>& virial,
            const std::vector<double>& coord,
            const std::vector<int>& atype,
            const std::vector<double>& box);

void DeepPotPT::get_type_map(std::string& type_map) {
  auto ret = module.run_method("get_type_map").toList();
  for (const torch::IValue& element : ret) {
      type_map += torch::str(element); // Convert each element to a string
      type_map += " "; // Add a space between elements 
  }
}

// forward to template method
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
    //TODO: atomic compute unsupported
  compute(ener, force, virial, coord, atype, box);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  //TODO: atomic compute unsupported
  compute(ener, force, virial, coord, atype, box);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
  //TODO: atomic compute unsupported
  compute(ener, force, virial, coord, atype, box, inlist,ago);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  //TODO: atomic compute unsupported
  compute(ener, force, virial, coord, atype, box, inlist,ago);
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<double>& force,
                                    std::vector<double>& virial,
                                    std::vector<double>& atom_energy,
                                    std::vector<double>& atom_virial,
                                    const int& nframes,
                                    const std::vector<double>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<double>& box,
                                    const std::vector<double>& fparam,
                                    const std::vector<double>& aparam) {
    throw;
  
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<float>& force,
                                    std::vector<float>& virial,
                                    std::vector<float>& atom_energy,
                                    std::vector<float>& atom_virial,
                                    const int& nframes,
                                    const std::vector<float>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<float>& box,
                                    const std::vector<float>& fparam,
                                    const std::vector<float>& aparam) {
  throw;
}
