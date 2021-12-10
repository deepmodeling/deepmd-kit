// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <iostream>
#include "attribute_def.h"

static void cum_sum(std::vector<int> &sec, const std::vector<int>& n_sel)
{
    sec.resize(n_sel.size() + 1);
    sec[0]=0;
    for(int ii=1; ii<(int)(sec.size());ii++)
    {
        sec[ii]=sec[ii-1]+n_sel[ii-1];
    }

}

/// Check the Targeting the IPU from TensorFlow document for
/// the API level required for the version of the Poplar SDK that you are using.
extern "C" {
  int32_t custom_op_api_level = 4;
}

/// This is an elementwise operation, so we tell the framework using the
/// Build_metadata function.
extern "C" void ProdEnvMatA_metadata(std::vector<std::int64_t>&            allocating_indices,
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

static void proc_rij(poplar::Graph&              graph, 
                     poplar::Tensor&             nlist_tensor,
                     poplar::Tensor&             numneigh_size_tensor,
                     poplar::Tensor&             npos_tensor,
                     poplar::Tensor&             ntype_tensor,
                     poplar::Tensor&             pos_loc_tensor,
                     poplar::Tensor&             fmt_idx_tensor,
                     poplar::Tensor&             rij_tensor,
                     int                         nloc,
                     int                         nall,
                     int                         nnei,
                     int                         max_nbor_size,
                     float                       cut_a,
                     float                       rcut_r,
                     float                       rcut_r_smth,
                     std::vector<int> const&     sec_a,
                     std::string const&          debugPrefix,
                     poplar::program::Sequence&  prog)
{
  (void)nall;
  (void)cut_a;
  (void)rcut_r_smth;
  poplar::ComputeSet     prod_init_rij_cs      = graph.addComputeSet(debugPrefix + std::string("/prod_init_rij_cs"));
  poplar::ComputeSet     prod_rij_cs           = graph.addComputeSet(debugPrefix + std::string("/prod_rij_cs"));

  poplar::Target const&  target          = graph.getTarget();
  unsigned int           num_tiles       = target.getNumTiles();
  int*                   tile_start      = new int[num_tiles];
  int*                   tile_count      = new int[num_tiles];
  int                    active_tile_cnt = 0;
  int                    tile_idx_last   = -1;
  memset(tile_start, 0, num_tiles * sizeof(int));
  memset(tile_count, 0, num_tiles * sizeof(int));
  //for (unsigned i = 0 ; i < kFrameCnt ; i ++)
  {
    for (unsigned j = 0; j < (unsigned)nloc; j ++)
    {
      int idx = (int)(((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nloc));
      //graph.setTileMapping(nlist_tensor[0][j],         idx);
      //graph.setTileMapping(numneigh_size_tensor[0][j], idx);
      //graph.setTileMapping(npos_tensor[0][j],          idx);
      //graph.setTileMapping(ntype_tensor[0][j],         idx);
      //graph.setTileMapping(pos_loc_tensor[0][j],       idx);
      graph.setTileMapping(fmt_idx_tensor[0][j],       idx);
      graph.setTileMapping(rij_tensor[0][j],           idx);
      if(tile_idx_last != idx)
      {
        active_tile_cnt ++;
        tile_start[idx] = j;
      }
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }
  poplar::Tensor sel_idx_tensor;
  if(active_tile_cnt > 0)
  {
    int sel_idx_len = 4 * max_nbor_size + 2 * nnei;
    sel_idx_tensor  = graph.addVariable(poplar::FLOAT, { (std::size_t)active_tile_cnt, (std::size_t)(sel_idx_len)}, "sel_idx");
  }

  std::string   zero_rij_vert_name = std::string("InitValueVertex<float>");
  std::string   zero_fmt_vert_name = std::string("InitValueVertex<int>");
  std::string   pod_rij_vert_name  = std::string("GenerateDistVertex<float>");
  if(poplar::HALF == npos_tensor.elementType())
  {
    zero_rij_vert_name = std::string("InitValueVertex<half>");
    pod_rij_vert_name  = std::string("GenerateDistVertex<half>");
  }
  active_tile_cnt = 0;
  for (unsigned i = 0; i < num_tiles; ++i)
  {
    if(0 == tile_count[i])
      continue;
    poplar::Tensor cur_nlist_tensor           =         nlist_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_numneigh_size_tensor   = numneigh_size_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_npos_tensor            =          npos_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_ntype_tensor           =         ntype_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_pos_loc_tensor         =       pos_loc_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_sel_idx_tensor         =       sel_idx_tensor[active_tile_cnt].flatten();
    poplar::Tensor cur_fmt_idx_tensor         =       fmt_idx_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_rij_tensor             =           rij_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::VertexRef  prod_init_rij_vertex   = graph.addVertex(prod_init_rij_cs, zero_rij_vert_name, 
                                                                {
                                                                  { "data_", cur_rij_tensor },
                                                                });
    graph.setTileMapping(prod_init_rij_vertex, i);
    graph.setInitialValue(prod_init_rij_vertex["value_"], 0.0);
    poplar::VertexRef  prod_init_fmt_idx_vertex   = graph.addVertex(prod_init_rij_cs, zero_fmt_vert_name, 
                                                                {
                                                                  { "data_", cur_fmt_idx_tensor },
                                                                });
    graph.setTileMapping(prod_init_fmt_idx_vertex, i);
    int   int_value   = -1;
    graph.setInitialValue(prod_init_fmt_idx_vertex["value_"], int_value);

    poplar::VertexRef  prod_rij_vertex        = graph.addVertex(prod_rij_cs, pod_rij_vert_name, 
                                                      {
                                                        { "nlist_",          cur_nlist_tensor },
                                                        { "nnumneigh_size_", cur_numneigh_size_tensor },
                                                        { "npos_",           cur_npos_tensor },
                                                        { "ntype_",          cur_ntype_tensor },
                                                        { "pos_loc_",        cur_pos_loc_tensor },
                                                        { "sel_idx_buf_",    cur_sel_idx_tensor },
                                                        { "fmt_idx_list_",   cur_fmt_idx_tensor },
                                                        { "rij_",            cur_rij_tensor }
                                                      });
    graph.setTileMapping(prod_rij_vertex, i);
    graph.setTileMapping(sel_idx_tensor[active_tile_cnt], i);
    
    graph.setInitialValue(prod_rij_vertex["max_nbor_size_"], (int)max_nbor_size);
    graph.setInitialValue(prod_rij_vertex["nnei_"],          (int)nnei);
    graph.setInitialValue(prod_rij_vertex["rcut_"],          rcut_r);
    graph.setInitialValue(prod_rij_vertex["sec_"],           sec_a);

    active_tile_cnt ++;
  }
  delete[] tile_start;
  delete[] tile_count;

  prog.add(poplar::program::Execute(prod_init_rij_cs));
  prog.add(poplar::program::Execute(prod_rij_cs));
}

extern "C" poplar::program::Program ProdEnvMatA(poplar::Graph&                      graph, 
                                                const std::vector<poplar::Tensor>&  inputs,
                                                std::vector<poplar::Tensor>&        outputs, 
                                                const std::string&                  attributes,
                                                const std::string&                  debugPrefix)
{
  (void)debugPrefix;
  int input_index = 0;
  const poplar::Tensor &coord_tensor        = inputs[input_index++];
  const poplar::Tensor &type_tensor_raw     = inputs[input_index++];
  const poplar::Tensor &natoms_tensor       = inputs[input_index++];
  const poplar::Tensor &box_tensor          = inputs[input_index++];//useless
  const poplar::Tensor &mesh_tensor         = inputs[input_index++];// delete
  const poplar::Tensor &avg_tensor_raw      = inputs[input_index++];
  const poplar::Tensor &std_tensor_raw      = inputs[input_index++];
  const poplar::Tensor &nloc_tensor         = inputs[input_index++];//modify
  const poplar::Tensor &nall_tensor         = inputs[input_index++];//modify
  const poplar::Tensor &ilist_tensor        = inputs[input_index++];//modify
  const poplar::Tensor &numneigh_tensor     = inputs[input_index++];//modify
  const poplar::Tensor &firstneigh_tensor   = inputs[input_index++];//modify
  const poplar::Tensor &type_nei_tensor_raw = inputs[input_index++];
  const poplar::Tensor &pos_nei_tensor_raw  = inputs[input_index++];

  (void)natoms_tensor;
  (void)box_tensor;
  (void)mesh_tensor;
  (void)ilist_tensor;

  Json::Value       json        = ParseAttributes(attributes);
  float             rcut_a      = json["rcut_a"].asFloat();
  float             rcut_r      = json["rcut_r"].asFloat();
  float             rcut_r_smth = json["rcut_r_smth"].asFloat();
  std::vector<int>  sel_a       = GetVectorFromJson(json["sel_a"]);
  std::vector<int>  sel_r       = GetVectorFromJson(json["sel_r"]);

  std::vector<int>  sec_a;
  std::vector<int>  sec_r;
  cum_sum(sec_a, sel_a);
  cum_sum(sec_r, sel_r);

  int  nloc           = (int)(nloc_tensor.shape()[0]);
  int  nall           = (int)(nall_tensor.shape()[0]);
  //int  ntypes         = (int)(natoms_tensor.shape()[0] - 2);
  int  nsamples       = (int)(coord_tensor.shape()[0]);
  int  nnei           = (int)(sec_a.back() + sec_r.back());
  int  ndescrpt       = (int)(sec_a.back() * 4) + sec_r.back() * 1;
  int  max_nbor_size  = firstneigh_tensor.numElements() / nloc;

  nsamples = 1;

  poplar::Target const&  target    = graph.getTarget();
  unsigned int           num_tiles = target.getNumTiles();
  poplar::Type           dtype     = coord_tensor.elementType();

  poplar::Tensor         npos_tensor           = pos_nei_tensor_raw.reshape({ (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)(max_nbor_size * 3)});
  poplar::Tensor         ntype_tensor          = type_nei_tensor_raw.reshape({ (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)max_nbor_size});
  poplar::Tensor         pos_loc_tensor        = coord_tensor.slice(0, 3 * nloc, 1).reshape({ (std::size_t)nsamples, (std::size_t)nloc, 3});
  poplar::Tensor         type_loc_tensor       = type_tensor_raw.slice(0, 1 * nloc, 1).reshape({ (std::size_t)nsamples, (std::size_t)nloc});
  poplar::Tensor         nlist_tensor          = firstneigh_tensor.reshape({ (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)max_nbor_size});
  poplar::Tensor         numneigh_size_tensor  = numneigh_tensor.reshape({ (std::size_t)nsamples, numneigh_tensor.dim(0)});
  poplar::Tensor         avg_tensor            = avg_tensor_raw.reshape({ (std::size_t)nsamples, avg_tensor_raw.numElements()});
  poplar::Tensor         std_tensor            = std_tensor_raw.reshape({ (std::size_t)nsamples, std_tensor_raw.numElements()});
  poplar::Tensor         fmt_idx_tensor        = graph.addVariable(poplar::INT,     { (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)nnei},           "fmt_idx");
  poplar::Tensor         rij_tensor            = graph.addVariable(poplar::FLOAT,   { (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)(nnei * 3)},     "rij");
  poplar::Tensor         env_mat_tensor        = graph.addVariable(poplar::FLOAT,   { (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)ndescrpt},       "env_mat");
  poplar::Tensor         env_mat_deriv_tensor  = graph.addVariable(poplar::FLOAT,   { (std::size_t)nsamples, (std::size_t)nloc, (std::size_t)(ndescrpt * 3)}, "env_mat_deriv");
/*
  {
    int*    tile_start      = new int[num_tiles];
    int*    tile_count      = new int[num_tiles];
    int     active_tile_cnt = 0;
    int     tile_idx_last   = -1;
    memset(tile_start, 0, num_tiles * sizeof(int));
    memset(tile_count, 0, num_tiles * sizeof(int));
    poplar::Tensor  pos_all_tensor   = coord_tensor.reshape({ (std::size_t)nsamples, (std::size_t)nall, 3});
    poplar::Tensor  type_all_tensor  = type_tensor_raw.reshape({ (std::size_t)nsamples, (std::size_t)nall});
    //for (unsigned i = 0 ; i < kFrameCnt ; i ++)
    {
      for (unsigned j = 0; j < nall; j ++)
      {
        int idx = (int)(((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nall));
        graph.setTileMapping(pos_all_tensor[0][j],    idx);
        graph.setTileMapping(type_all_tensor[0][j],   idx);
        if(tile_idx_last != idx)
          tile_start[idx] = j;
        tile_count[idx] += 1;
        tile_idx_last    = idx;
      }
    }
    delete[] tile_start;
    delete[] tile_count;
  }
*/
  poplar::program::Sequence prog;
  proc_rij(graph, 
           nlist_tensor,
           numneigh_size_tensor,
           npos_tensor,
           ntype_tensor,
           pos_loc_tensor,
           fmt_idx_tensor,
           rij_tensor,
           nloc,
           nall,
           nnei,
           max_nbor_size,
           rcut_a,
           rcut_r,
           rcut_r_smth,
           sec_a,
           debugPrefix,
           prog);

  int*    tile_start      = new int[num_tiles];
  int*    tile_count      = new int[num_tiles];
  int     active_tile_cnt = 0;
  int     tile_idx_last   = -1;
  memset(tile_start, 0, num_tiles * sizeof(int));
  memset(tile_count, 0, num_tiles * sizeof(int));
  //for (unsigned i = 0 ; i < kFrameCnt ; i ++)
  {
    for (unsigned j = 0; j < (unsigned)nloc; j ++)
    {
      int idx = (int)(((unsigned long long)j * (unsigned long long)num_tiles) / ((unsigned long long)nloc));
      //graph.setTileMapping(type_loc_tensor[0][j],      idx);
      graph.setTileMapping(fmt_idx_tensor[0][j],       idx);
      graph.setTileMapping(rij_tensor[0][j],           idx);
      graph.setTileMapping(env_mat_tensor[0][j],       idx);
      graph.setTileMapping(env_mat_deriv_tensor[0][j], idx);
      if(tile_idx_last != idx)
      {
        active_tile_cnt ++;
        tile_start[idx] = j;
      }
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }
  poplar::Tensor        avg_tensor_all;
  poplar::Tensor        std_tensor_all;
  if(active_tile_cnt > 0)
  {
    avg_tensor_all   = avg_tensor.expand({1});
    avg_tensor_all   = avg_tensor_all.broadcast((std::size_t)active_tile_cnt, 1);

    std_tensor_all   = std_tensor.expand({1});
    std_tensor_all   = std_tensor_all.broadcast((std::size_t)active_tile_cnt, 1);
  }

  poplar::ComputeSet    prod_init_em_cs       = graph.addComputeSet(debugPrefix + std::string("/prod_init_em_cs"));
  poplar::ComputeSet    prod_init_em_deriv_cs = graph.addComputeSet(debugPrefix + std::string("/prod_init_em_deriv_cs"));
  poplar::ComputeSet    prod_env_mat_cs       = graph.addComputeSet(debugPrefix + std::string("/prod_env_mat_cs"));
  std::string           em_zero_vert_name     = "InitValueVertex<float>";
  if(poplar::HALF == dtype)
    em_zero_vert_name = "InitValueVertex<half>";
  active_tile_cnt = 0;
  for (unsigned i = 0; i < num_tiles; ++i)
  {
    if(0 == tile_count[i])
      continue;

    poplar::Tensor cur_env_mat_tensor         =        env_mat_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor cur_env_mat_deriv_tensor   =  env_mat_deriv_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::VertexRef     prod_init_em_vertex = graph.addVertex(prod_init_em_cs, em_zero_vert_name, 
                                                                {
                                                                  { "data_", cur_env_mat_tensor },
                                                                });
    graph.setTileMapping(prod_init_em_vertex, i);
    graph.setInitialValue(prod_init_em_vertex["value_"], 0.0);

    poplar::VertexRef     prod_init_em_deriv_vertex   = graph.addVertex(prod_init_em_deriv_cs, em_zero_vert_name, 
                                                                {
                                                                  { "data_", cur_env_mat_deriv_tensor },
                                                                });
    graph.setTileMapping(prod_init_em_deriv_vertex, i);
    graph.setInitialValue(prod_init_em_deriv_vertex["value_"], 0.0);

    poplar::Tensor     cur_type_tensor          =      type_loc_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor     cur_fmt_idx_tensor       =       fmt_idx_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor     cur_rij_tensor           =           rij_tensor[0].slice(tile_start[i], tile_start[i] + tile_count[i], 0).flatten();
    poplar::Tensor     cur_avg_tensor_all       = avg_tensor_all[0][active_tile_cnt].flatten();
    poplar::Tensor     cur_std_tensor_all       = std_tensor_all[0][active_tile_cnt].flatten();
    poplar::VertexRef  prod_env_mat_vertex      = graph.addVertex(prod_env_mat_cs, "ProdEnvMatVertex<float>", 
                                                                {
                                                                  { "type_loc_",       cur_type_tensor},
                                                                  { "fmt_idx_list_",   cur_fmt_idx_tensor },
                                                                  { "rij_",            cur_rij_tensor },
                                                                  { "avg_",            cur_avg_tensor_all },
                                                                  { "std_",            cur_std_tensor_all },
                                                                  { "env_mat_",        cur_env_mat_tensor },
                                                                  { "env_mat_deriv_",  cur_env_mat_deriv_tensor },
                                                                });
    graph.setTileMapping(prod_env_mat_vertex, i);
    graph.setTileMapping(avg_tensor_all[0][active_tile_cnt], i);
    graph.setTileMapping(std_tensor_all[0][active_tile_cnt], i);
    graph.setInitialValue(prod_env_mat_vertex["nnei_"],        (int)nnei);
    graph.setInitialValue(prod_env_mat_vertex["rcut_smooth_"], rcut_r_smth);
    graph.setInitialValue(prod_env_mat_vertex["rcut_"],        rcut_r);
    graph.setInitialValue(prod_env_mat_vertex["sec_"],         sec_a);
    active_tile_cnt ++;
  }
  delete[] tile_start;
  delete[] tile_count;

  prog.add(poplar::program::Execute(prod_init_em_cs));
  prog.add(poplar::program::Execute(prod_init_em_deriv_cs));
  prog.add(poplar::program::Execute(prod_env_mat_cs));

  env_mat_tensor       =       env_mat_tensor.reshape({(std::size_t)nloc, (std::size_t)(nnei*4)});
  env_mat_deriv_tensor = env_mat_deriv_tensor.reshape({(std::size_t)nloc, (std::size_t)(nnei*4*3)});
  rij_tensor           =           rij_tensor.reshape({(std::size_t)nloc, (std::size_t)(nnei*3)});
  fmt_idx_tensor       =       fmt_idx_tensor.reshape({(std::size_t)nloc, (std::size_t)nnei});

  outputs.push_back(env_mat_tensor);
  outputs.push_back(env_mat_deriv_tensor);
  outputs.push_back(rij_tensor);
  outputs.push_back(fmt_idx_tensor);
  return prog;
}
