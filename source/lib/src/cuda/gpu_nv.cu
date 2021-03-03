#include "gpu_nv.h"
#include "device.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

// common part of prod_env_mat
template <
    typename    Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
__global__ void BlockSortKernel(
    Key * d_in,
    Key * d_out)                // Tile of output
{   
  enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockRadixSort type for our thread block
  typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage        load;
    typename BlockRadixSortT::TempStorage   sort;
  } temp_storage;
  // Per-thread tile items
  Key items[ITEMS_PER_THREAD];
  // Our current block's offset
  int block_offset = blockIdx.x * TILE_SIZE;
  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
  // Barrier for smem reuse
  __syncthreads();
  // Sort keys
  BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);
  // Store output in striped fashion
  cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
}

template __global__ void BlockSortKernel<int_64, 128, 8>(int_64 * d_in, int_64 * d_out);
template __global__ void BlockSortKernel<int_64, 256, 8>(int_64 * d_in, int_64 * d_out);
template __global__ void BlockSortKernel<int_64, 256, 16>(int_64 * d_in, int_64 * d_out);
