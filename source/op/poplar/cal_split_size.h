#ifndef __SPLIT_SIZE_H__
#define __SPLIT_SIZE_H__

#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <vector>
#include <stdio.h>

static inline std::vector<int> cal_split_size(poplar::Target const& target, 
                                              poplar::Type          dtype,    
                                              int                   ele_cnt, 
                                              int                   min_cell_size,
                                              float                 split_ratio)
{
  int    element_size         = poplar::FLOAT == dtype ? 4 : 2;
  int    max_bytes_per_tile   = target.getBytesPerTile();
  int    max_payload_per_tile = min_cell_size * ((((int)(max_bytes_per_tile * split_ratio)) + min_cell_size - 1) / min_cell_size);
  int    total_payload        = ele_cnt * element_size;
  int    blk_cnt              = (total_payload + max_payload_per_tile - 1) / max_payload_per_tile;
  std::vector<int>  blk_size(blk_cnt);
  int               elapsed_size       = 0;
  for(int i = 0 ; i < blk_cnt - 1; i ++)
  {
    blk_size[i]   = (max_payload_per_tile / element_size);
    elapsed_size += max_payload_per_tile;
  }
  blk_size[blk_cnt - 1] = (total_payload - elapsed_size) / element_size;
/*
  printf("blk info: %d, %d, %d, %d\n", total_payload, max_bytes_per_tile, max_payload_per_tile, blk_cnt);
  for(int i = 0 ; i < blk_cnt ; i ++)
    printf("%d, ", blk_size[i]);
  printf("\n");
*/
  return blk_size;
}

#endif