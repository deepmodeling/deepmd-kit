#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <iostream>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include "cal_split_size.h"

extern "C" void ProdForceSeA_metadata(std::vector<std::int64_t>&            allocating_indices,
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

extern "C" poplar::program::Program ProdForceSeA(poplar::Graph&                      graph, 
                                                 const std::vector<poplar::Tensor>&  inputs,
                                                 std::vector<poplar::Tensor>&        outputs, 
                                                 const std::string&                  attributes,
                                                 const std::string&                  debugPrefix)
{
  (void)attributes;
  (void)debugPrefix;
  int   input_index=0;
  const poplar::Tensor& net_deriv_tensor_raw  = inputs[input_index++];
  const poplar::Tensor& env_deriv_tensor_raw  = inputs[input_index++];
  const poplar::Tensor& nlist_tensor_raw      = inputs[input_index++];
  const poplar::Tensor& nloc_tensor           = inputs[input_index++];
  const poplar::Tensor& nall_tensor           = inputs[input_index++];

  unsigned int nall     = nall_tensor.shape()[0];
  unsigned int nloc     = nloc_tensor.shape()[0];
  unsigned int nframes  = net_deriv_tensor_raw.shape()[0];
  //unsigned int nnei     = nlist_tensor_raw.shape()[1] / nloc;
  int          nnei     = nlist_tensor_raw.shape()[1];
  poplar::Tensor net_deriv_tensor = net_deriv_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei * 4) });
  poplar::Tensor env_deriv_tensor = env_deriv_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc, (std::size_t)(nnei * 4 * 3) });
  poplar::Tensor nlist_tensor     = nlist_tensor_raw.reshape({ (std::size_t)nframes, (std::size_t)nloc,     (std::size_t)(nnei) });

  popops::addCodelets(graph);

  poplar::Target const&  target    = graph.getTarget();
  unsigned int           num_tiles = target.getNumTiles();
  poplar::Type           dtype     = net_deriv_tensor.elementType();

  std::vector<int>       blk_size  = cal_split_size(target, dtype, 3 * nall, 24, 1.0f);

  poplar::Tensor    force_2nd_tensor   = graph.addVariable(dtype, { (std::size_t)nframes, (std::size_t)(nall * 3)}, debugPrefix + std::string("/2nd_force"));
  int               active_tile_cnt    = 0;
  int*              tile_start         = new int[num_tiles];
  int*              tile_count         = new int[num_tiles];
  int               tile_idx_last      = -1;
  memset(tile_start, 0, num_tiles * sizeof(int));
  memset(tile_count, 0, num_tiles * sizeof(int));
  poplar::Tensor     force_2nd_tensor_reshape = force_2nd_tensor.reshape({(std::size_t)nframes, (std::size_t)nall, 3});
  for (unsigned j = 0; j < (unsigned)nall; j ++)
  {
    int idx = ((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nall);
    graph.setTileMapping(force_2nd_tensor_reshape[0][j], idx);
  } 
  //for (unsigned i = 0 ; i < nframes ; i ++)
  {
    for (unsigned j = 0; j < nloc; j ++)
    {
      int idx = (int)(((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nloc));
      //graph.setTileMapping(net_deriv_tensor[0][j],  (unsigned)idx);
      //graph.setTileMapping(env_deriv_tensor[0][j],  (unsigned)idx);
      //graph.setTileMapping(nlist_tensor[0][j],      (unsigned)idx);
      if(tile_idx_last != idx)
      {
        active_tile_cnt ++;
        tile_start[idx] = j;
      }
        
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }
  poplar::Tensor    force_2nd_tensor_bc;  
  if(active_tile_cnt > 0)
    force_2nd_tensor_bc = graph.addVariable(dtype, { (std::size_t)nframes, (std::size_t)active_tile_cnt, (std::size_t)blk_size[0]}, debugPrefix + std::string("/2nd_force_bc"));

  std::string  force2nd_vertex_name = "ProdForceSecondVertex";
  std::string  init_val_vertex_name = "InitValueVertex";
  std::string  force1st_vertex_name = "ProdForceFirstVertex";
  if(poplar::FLOAT == dtype)
  {
    force2nd_vertex_name += "<float>";
    init_val_vertex_name += "<float>";
    force1st_vertex_name += "<float>";
  }
  else
  {
    force2nd_vertex_name += "<half>";
    init_val_vertex_name += "<half>";
    force1st_vertex_name += "<half>";
  }
  poplar::program::Sequence prog = poplar::program::Sequence();
  int  proc_start = 0;
  for(int k = 0 ; k < (int)(blk_size.size()) ; k ++)
  {
    poplar::Tensor  force_2nd_tensor_proc   = force_2nd_tensor_bc;
    poplar::Tensor  force_2nd_tensor_dst    = force_2nd_tensor.slice((unsigned int)(proc_start * 3), (unsigned int)(proc_start * 3 + blk_size[k]), 1);
    if((int)(force_2nd_tensor_bc.dim(2)) != blk_size[k])
      force_2nd_tensor_proc = force_2nd_tensor_bc.slice(0, (unsigned int)blk_size[k], 2);
    
    std::string         force_zero_cs_name = debugPrefix + std::string("/ProdFroceZeroCS") + std::to_string(k);
    std::string         force_2nd_cs_name  = debugPrefix + std::string("/ProdFroce2ndCS")  + std::to_string(k);
    poplar::ComputeSet  prod_force_zero_cs = graph.addComputeSet(force_zero_cs_name);
    poplar::ComputeSet  prod_force_2nd_cs  = graph.addComputeSet(force_2nd_cs_name);

    active_tile_cnt = 0;
    for (unsigned i = 0; i < num_tiles; ++i)
    {
      if(0 == tile_count[i])
        continue;
      poplar::Tensor    cur_bc_force_tensor = force_2nd_tensor_proc[0][active_tile_cnt].flatten();
      poplar::VertexRef prod_zero_vertex    = graph.addVertex(prod_force_zero_cs, init_val_vertex_name, 
                                                        {
                                                          {"data_",      cur_bc_force_tensor},
                                                        });
      graph.setTileMapping(prod_zero_vertex, i);
      graph.setInitialValue(prod_zero_vertex["value_"], 0.0f);
      graph.setTileMapping(force_2nd_tensor_proc[0][active_tile_cnt],  i);

      poplar::Tensor cur_net_deriv_tensor = net_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::Tensor cur_env_deriv_tensor = env_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
      poplar::Tensor cur_nlist_tensor     =     nlist_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();

      poplar::VertexRef  prod_force_2nd_vertex = graph.addVertex(prod_force_2nd_cs, force2nd_vertex_name, 
                                                        {
                                                          {"net_deriv_",  cur_net_deriv_tensor},
                                                          {"env_deriv_",  cur_env_deriv_tensor},
                                                          {"nlist_",      cur_nlist_tensor},
                                                          {"force_",      cur_bc_force_tensor},
                                                        });
      graph.setTileMapping(prod_force_2nd_vertex, i);
      graph.setInitialValue(prod_force_2nd_vertex["nloc_"],  (int)nloc);
      graph.setInitialValue(prod_force_2nd_vertex["nnei_"],  (int)nnei);
      graph.setInitialValue(prod_force_2nd_vertex["start_"], (int)proc_start);
      active_tile_cnt ++;
    }
    prog.add(poplar::program::Sequence(poplar::program::Execute(prod_force_zero_cs)));
    prog.add(poplar::program::Sequence(poplar::program::Execute(prod_force_2nd_cs)));

    std::string reduce_name = debugPrefix + std::string("/reduce_sum") + std::to_string(k);
    popops::reduceWithOutput(graph,
                             force_2nd_tensor_proc,
                             force_2nd_tensor_dst,
                             { 1 },
                             {popops::Operation::ADD},
                             prog,
                             reduce_name);
    proc_start += (blk_size[k] / 3);
  }

  for (unsigned j = 0; j < nloc; j ++)
  {
    int idx = (int)(((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nloc));
    graph.setTileMapping(force_2nd_tensor_reshape[0][j], (unsigned)idx);
  }

  std::string         force_1st_cs_name  = debugPrefix + std::string("/ProdFroce1stCS");
  poplar::ComputeSet  prod_force_1st_cs  = graph.addComputeSet(force_1st_cs_name);
  for (unsigned i = 0; i < num_tiles; ++i)
  {
    if(0 == tile_count[i])
      continue;
    poplar::Tensor     cur_net_deriv_tensor = net_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
    poplar::Tensor     cur_env_deriv_tensor = env_deriv_tensor.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
    poplar::Tensor     cur_force_tensor     = force_2nd_tensor_reshape.slice(tile_start[i], tile_start[i] + tile_count[i], 1).flatten();
    poplar::VertexRef  prod_force_1st_vertex = graph.addVertex(prod_force_1st_cs, force1st_vertex_name, 
                                                      {
                                                        {"net_deriv_",  cur_net_deriv_tensor},
                                                        {"env_deriv_",  cur_env_deriv_tensor},
                                                        {"force_",      cur_force_tensor},
                                                      });
    graph.setTileMapping(prod_force_1st_vertex, i);
    graph.setInitialValue(prod_force_1st_vertex["nnei_"],  (int)nnei);
  }
  poplar::program::Program prog_1st = poplar::program::Sequence(poplar::program::Execute(prod_force_1st_cs));
  prog.add(prog_1st);

  delete tile_start;
  delete tile_count;

  outputs.resize(1);
  force_2nd_tensor = force_2nd_tensor.flatten();
  force_2nd_tensor = force_2nd_tensor.expand({0});
  outputs[0]       = force_2nd_tensor;

  return prog ;
}