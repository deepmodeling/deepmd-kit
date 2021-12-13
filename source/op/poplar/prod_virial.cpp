
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <iostream>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include "cal_split_size.h"

extern "C" void ProdVirialSeA_metadata(std::vector<std::int64_t>&            allocating_indices,
                                       std::map<std::int64_t, std::int64_t>& input_to_output_tensor_aliasing,
                                       bool&                                 is_elementwise,
                                       bool&                                 is_stateless,
                                       bool&                                 is_hashable,
                                       std::uint32_t                         num_inputs) 
{
  (void)allocating_indices;
  (void)input_to_output_tensor_aliasing;
  (void)is_stateless;
  (void)is_hashable;
  (void)num_inputs;
  is_elementwise = false;
}

extern "C" poplar::program::Program ProdVirialSeA(poplar::Graph&                      graph, 
                                                  const std::vector<poplar::Tensor>&  inputs,
                                                  std::vector<poplar::Tensor>&        outputs, 
                                                  const std::string&                  attributes,
                                                  const std::string&                  debugPrefix)
{
  (void)attributes;
  (void)debugPrefix;
  int   input_index = 0;
  const poplar::Tensor &net_deriv_tensor_raw = inputs[input_index++];
  const poplar::Tensor &env_deriv_tensor_raw = inputs[input_index++];
  const poplar::Tensor &rij_tensor_raw       = inputs[input_index++];
  const poplar::Tensor &nlist_tensor_raw     = inputs[input_index++];
  const poplar::Tensor &nloc_tensor          = inputs[input_index++];
  const poplar::Tensor &nall_tensor          = inputs[input_index++];

  int nloc     = nloc_tensor.shape()[0];
  int nall     = nall_tensor.shape()[0];
  //int ndescrpt = net_deriv_tensor_raw.shape()[1] / nloc;
  int nframes  = net_deriv_tensor_raw.shape()[0];
  //int nnei     = nlist_tensor_raw.shape()[1] / nloc;
  int nnei     = nlist_tensor_raw.shape()[1];

  poplar::Tensor net_deriv_tensor   = net_deriv_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei * 4) });
  poplar::Tensor env_deriv_tensor   = env_deriv_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei * 4 * 3) });
  poplar::Tensor rij_tensor         =       rij_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei * 3) });
  poplar::Tensor nlist_tensor       =     nlist_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei) });

  popops::addCodelets(graph);

  poplar::Target const&  target          = graph.getTarget();
  unsigned int           num_tiles       = target.getNumTiles();
  poplar::Type           dtype           = net_deriv_tensor.elementType();
  
  int                    active_tile_cnt = 0;
  int*                   tile_start      = new int[num_tiles];
  int*                   tile_count      = new int[num_tiles];
  int                    tile_idx_last   = -1;
  memset(tile_start, 0, num_tiles * sizeof(int));
  memset(tile_count, 0, num_tiles * sizeof(int));

  poplar::Tensor atom_virial_tensor         = graph.addVariable(dtype, {(std::size_t)nframes, (std::size_t)(nall * 9)}, debugPrefix + std::string("/atom_virial"));
  poplar::Tensor atom_virial_tensor_reshape = atom_virial_tensor.reshape({(std::size_t)nframes, (std::size_t)nall, 9});
  poplar::program::Sequence prog = poplar::program::Sequence();
  for (unsigned j = 0; j < (unsigned)nall; j ++)
  {
    int idx = ((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nall);
    graph.setTileMapping(atom_virial_tensor_reshape[0][j], idx);
  }

  //for (unsigned i = 0 ; i < (unsigned)nframes ; i ++)
  {
    for (unsigned j = 0; j < (unsigned)nloc; j ++)
    {
      int idx = ((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nloc);
      //graph.setTileMapping(net_deriv_tensor[0][j],  idx);
      //graph.setTileMapping(env_deriv_tensor[0][j],  idx);
      //graph.setTileMapping(rij_tensor[0][j],        idx);
      //graph.setTileMapping(nlist_tensor[0][j],      idx);
      if(tile_idx_last != idx)
      {
        active_tile_cnt ++;
        tile_start[idx] = j;
      }
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }

  std::vector<int>  blk_size   = cal_split_size(target, dtype, 3 * nall, 72, 1.0f);

  poplar::Tensor virial_tensor_bc;
  poplar::Tensor atom_virial_tensor_bc;
  if(active_tile_cnt > 0)
  {
    virial_tensor_bc      = graph.addVariable(dtype, {(std::size_t)nframes, (std::size_t)active_tile_cnt, 9}, debugPrefix + std::string("/virial_bc"));
    atom_virial_tensor_bc = graph.addVariable(dtype, {(std::size_t)nframes, (std::size_t)active_tile_cnt, (std::size_t)blk_size[0]}, debugPrefix + std::string("/atom_virial_bc"));
  }

  std::string init_vertex_name = "InitValueVertex";
  std::string prod_vertex_name = "ProdVirialVertex";
  if(poplar::FLOAT == dtype)
  {
    init_vertex_name += "<float>";
    prod_vertex_name += "<float>";
  }
  else
  {
    init_vertex_name += "<half>";
    prod_vertex_name += "<half>";
  }

  int  proc_start = 0;
  for(int k = 0 ; k < (int)(blk_size.size()) ; k ++)
  {
    poplar::Tensor  atom_virial_tensor_proc = atom_virial_tensor_bc;
    poplar::Tensor  atom_virial_tensor_dst  = atom_virial_tensor.slice((unsigned int)(proc_start * 9), (unsigned int)(proc_start * 9 + blk_size[k]), 1);
    if((int)(atom_virial_tensor_bc.dim(2)) != blk_size[k])
    {
      atom_virial_tensor_proc = atom_virial_tensor_bc.slice(0, (unsigned int)blk_size[k], 2);
    }
    std::string        init_cs_name        = debugPrefix + std::string("/ProdVirialInitCS") + std::to_string(k);
    std::string        virial_cs_name      = debugPrefix + std::string("/ProdVirialCS")     + std::to_string(k);
    poplar::ComputeSet prod_init_virial_cs = graph.addComputeSet(init_cs_name);
    poplar::ComputeSet prod_virial_cs      = graph.addComputeSet(virial_cs_name);
    active_tile_cnt                        = 0;
    for (unsigned i = 0; i < num_tiles; ++i)
    {
      if(0 == tile_count[i])
        continue;

      poplar::Tensor cur_virial_tensor       =      virial_tensor_bc[0][active_tile_cnt].flatten();
      poplar::Tensor cur_atom_virial_tensor  = atom_virial_tensor_proc[0][active_tile_cnt].flatten();
      if(0 == k)
      {
        poplar::VertexRef  prod_virial_init_vertex  = graph.addVertex(prod_init_virial_cs, init_vertex_name, 
                                                          {
                                                            {"data_",      cur_virial_tensor},
                                                          });
        graph.setTileMapping(prod_virial_init_vertex, i);
        graph.setInitialValue(prod_virial_init_vertex["value_"], 0.0f);
      }

      poplar::VertexRef  prod_atom_virial_init_vertex  = graph.addVertex(prod_init_virial_cs, init_vertex_name, 
                                                        {
                                                          {"data_",      cur_atom_virial_tensor},
                                                        });
      graph.setTileMapping(prod_atom_virial_init_vertex, i);
      graph.setInitialValue(prod_atom_virial_init_vertex["value_"], 0.0f);

      graph.setTileMapping(virial_tensor_bc[0][active_tile_cnt],      i);
      graph.setTileMapping(atom_virial_tensor_proc[0][active_tile_cnt], i);

      poplar::Tensor cur_net_deriv_tensor    =      net_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::Tensor cur_env_deriv_tensor    =      env_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::Tensor cur_rij_tensor          =            rij_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::Tensor cur_nlist_tensor        =          nlist_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::VertexRef  prod_virial_vertex  = graph.addVertex(prod_virial_cs, prod_vertex_name, 
                                                        {
                                                          {"net_deriv_",   cur_net_deriv_tensor},
                                                          {"env_deriv_",   cur_env_deriv_tensor},
                                                          {"rij_",         cur_rij_tensor},
                                                          {"nlist_",       cur_nlist_tensor},
                                                          {"virial_",      cur_virial_tensor},
                                                          {"atom_virial_", cur_atom_virial_tensor},
                                                        });
      graph.setTileMapping(prod_virial_vertex, i);
      graph.setInitialValue(prod_virial_vertex["nloc_"],  (int)nloc);
      graph.setInitialValue(prod_virial_vertex["nnei_"],  (int)nnei);
      graph.setInitialValue(prod_virial_vertex["start_"], proc_start);

      active_tile_cnt ++;
    }
    prog.add(poplar::program::Execute(prod_init_virial_cs));
    prog.add(poplar::program::Execute(prod_virial_cs));

    std::string reduce_name_1 = debugPrefix + std::string("/atom_virial_reduce_sum") + std::to_string(k);
    popops::reduceWithOutput(graph,
                             atom_virial_tensor_proc,
                             atom_virial_tensor_dst,
                             { 1 },
                             {popops::Operation::ADD},
                             prog,
                             reduce_name_1.c_str());

    proc_start += (blk_size[k] / 9);
  }
  poplar::Tensor virial_tensor = popops::reduce(graph,
                            virial_tensor_bc,
                            { 1 },
                            {popops::Operation::ADD},
                            prog,
                            debugPrefix + std::string("/virial_reduce_sum"));
  delete tile_start;
  delete tile_count;
  outputs.resize(2);
  outputs[0] = virial_tensor;
  outputs[1] = atom_virial_tensor;

  return prog ;
}