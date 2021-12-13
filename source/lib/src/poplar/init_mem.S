#ifdef __IPU__

#ifndef __INIT_MEMORY_ASM_COMMON__
#define __INIT_MEMORY_ASM_COMMON__

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define  RPT_BITS                     11

#define  FP32_PROC_KERNEL_BITS        4
#define  FP32_PROC_KERNEL_SIZE        (1 << FP32_PROC_KERNEL_BITS)  //16
#define  FP32_PROC_BLK_BITS           (FP32_PROC_KERNEL_BITS + RPT_BITS)
#define  FP32_PROC_BLK_LEN            (1 << FP32_PROC_BLK_BITS)

#define  FP16_PROC_KERNEL_BITS        5
#define  FP16_PROC_KERNEL_SIZE        (1 << FP16_PROC_KERNEL_SIZE)  //32
#define  FP16_PROC_BLK_BITS           (FP16_PROC_KERNEL_BITS + RPT_BITS)
#define  FP16_PROC_BLK_LEN            (1 << FP16_PROC_BLK_BITS)

#define  INT32_PROC_KERNEL_BITS       4
#define  INT32_PROC_KERNEL_SIZE       (1 << INT32_PROC_KERNEL_BITS)  //16
#define  INT32_PROC_BLK_BITS          (INT32_PROC_KERNEL_BITS + RPT_BITS)
#define  INT32_PROC_BLK_LEN           (1 << INT32_PROC_BLK_BITS)

.macro INIT_MEM_KERNEL_PROC
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
  {
    st64step        $a0:1,          $m15,              $m0+=,             1
    fnop
  }
.endm

	.text
	.allow_optimizations
	.file	"4"
	.section	.text._Z14InitMemoryFp32Pfif,"ax",@progbits
	.globl	_Z14InitMemoryFp32Pfif           # -- Begin function _Z14InitMemoryFp32Pfif
	.p2align	2
	.type	_Z14InitMemoryFp32Pfif,@function
_Z14InitMemoryFp32Pfif:                  # @_Z14InitMemoryFp32Pfif
._Z14InitMemoryFp32Pfif_start:
  mov               $a1,              $a0
  mov               $m2,              $m1
  shr               $m3,              $m1,                FP32_PROC_BLK_BITS   //shrink_blk_cnt
  brz               $m3,              __InitMemoryFp32P_blk_loop_stop
  sub               $m3,              $m3,                1
  ldconst           $m4,              FP32_PROC_BLK_LEN
__InitMemoryFp32P_blk_loop_start:
.align 8
  {
    rpt             (1 << RPT_BITS), ((__init_mem_fp32_internal_1st_loop_end - __init_mem_fp32_internal_1st_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp32_internal_1st_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_fp32_internal_1st_loop_end:
  sub               $m2,              $m2,                $m4
  brnzdec           $m3,              __InitMemoryFp32P_blk_loop_start
__InitMemoryFp32P_blk_loop_stop:
  brz               $m2,              __init_mem_fp32_exit
  shr               $m3,              $m2,                FP32_PROC_KERNEL_BITS   //kernel_cnt
  shl               $m1,              $m3,                FP32_PROC_KERNEL_BITS   //shrink_proc_len
.align 8
  {
    rpt             $m3,              ((__init_mem_fp32_internal_2nd_loop_end - __init_mem_fp32_internal_2nd_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp32_internal_2nd_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_fp32_internal_2nd_loop_end:
  sub               $m2,              $m2,               $m1
  brz               $m2,              __init_mem_fp32_exit
.align 8
  {
    rpt             $m2,              ((__init_mem_fp32_tail_loop_end - __init_mem_fp32_tail_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp32_tail_loop_start:
  {
    st32step        $a0,              $m15,              $m0+=,             1
    fnop
  }
__init_mem_fp32_tail_loop_end:
__init_mem_fp32_exit:
  br                $lr
._Z14InitMemoryFp32Pfif_end:
	.size	_Z14InitMemoryFp32Pfif, ._Z14InitMemoryFp32Pfif_end-_Z14InitMemoryFp32Pfif
	.section	.stack_sizes,"o",@progbits,.text._Z14InitMemoryFp32Pfif
	.long	._Z14InitMemoryFp32Pfif_start
  .byte	0
	.section	.text._Z14InitMemoryFp32Pfif,"ax",@progbits
                                        # -- End function

	.section	.text._Z14InitMemoryFp16PDhiDh,"ax",@progbits
	.globl	_Z14InitMemoryFp16PDhiDh        # -- Begin function _Z14InitMemoryFp16PDhiDh
	.p2align	2
	.type	_Z14InitMemoryFp16PDhiDh,@function
_Z14InitMemoryFp16PDhiDh:               # @_Z14InitMemoryFp16PDhiDh
.Z14InitMemoryFp16PDhiDh_start:
  mov	              $a1,              $a0
  sort4x16lo        $a1,              $a0,                $a0
  mov               $a0,              $a1

  mov               $m2,              $m1
  shr               $m3,              $m1,                FP16_PROC_BLK_BITS   //shrink_blk_cnt
  brz               $m3,              __InitMemoryFp16_blk_loop_stop
  sub               $m3,              $m3,                1

  ldconst           $m4,              FP16_PROC_BLK_LEN
__InitMemoryFp16_blk_loop_start:
.align 8
  {
    rpt             (1 << RPT_BITS), ((__init_mem_fp16_internal_1st_loop_end - __init_mem_fp16_internal_1st_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp16_internal_1st_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_fp16_internal_1st_loop_end:
  sub               $m2,              $m2,                $m4
  brnzdec           $m3,              __InitMemoryFp16_blk_loop_start
__InitMemoryFp16_blk_loop_stop:
  brz               $m2,              __init_mem_fp16_exit
  shr               $m3,              $m2,                FP16_PROC_KERNEL_BITS   //kernel_cnt
  shl               $m1,              $m3,                FP16_PROC_KERNEL_BITS   //shrink_proc_len
.align 8
  {
    rpt             $m3,              ((__init_mem_fp16_internal_2nd_loop_end - __init_mem_fp16_internal_2nd_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp16_internal_2nd_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_fp16_internal_2nd_loop_end:
  sub               $m2,              $m2,               $m1
  brz               $m2,              __init_mem_fp16_exit
  shr               $m2,              $m2,               1
.align 8
  {
    rpt             $m2,              ((__init_mem_fp16_tail_loop_end - __init_mem_fp16_tail_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_fp16_tail_loop_start:
  {
    st32step        $a0,              $m15,              $m0+=,             1
    fnop
  }
__init_mem_fp16_tail_loop_end:
__init_mem_fp16_exit:
   br          $lr
.Z14InitMemoryFp16PDhiDh_stop:
	.size	_Z14InitMemoryFp16PDhiDh, .Z14InitMemoryFp16PDhiDh_stop-_Z14InitMemoryFp16PDhiDh
	.section	.stack_sizes,"o",@progbits,.text._Z14InitMemoryFp16PDhiDh
	.long	.Z14InitMemoryFp16PDhiDh_start
  .byte	48
	.section	.text._Z14InitMemoryFp16PDhiDh,"ax",@progbits
                                        # -- End function

	.section	.text._Z15InitMemoryInt32Piii,"ax",@progbits
	.globl	_Z15InitMemoryInt32Piii         # -- Begin function _Z15InitMemoryInt32Piii
	.p2align	2
	.type	_Z15InitMemoryInt32Piii,@function
_Z15InitMemoryInt32Piii:                # @_Z15InitMemoryInt32Piii
.Z15InitMemoryInt32Piii_start:
  brz               $m1,              __init_mem_int32_exit
  st32              $m2,              $m0,                0
  ld32              $a0,              $m0,                0
  mov               $a1,              $a0

  ldconst           $m4,              INT32_PROC_BLK_LEN

  mov               $m2,              $m1
  shr               $m3,              $m1,                INT32_PROC_BLK_BITS   //shrink_blk_cnt
  brz               $m3,              __InitMemoryInt32_blk_loop_stop
  sub               $m3,              $m3,                1
__InitMemoryInt32_blk_loop_start:
.align 8
  {
    rpt             (1 << RPT_BITS),  ((__init_mem_int32_internal_1st_loop_end - __init_mem_int32_internal_1st_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_int32_internal_1st_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_int32_internal_1st_loop_end:
  sub               $m2,              $m2,                $m4
  brnzdec           $m3,              __InitMemoryInt32_blk_loop_start
__InitMemoryInt32_blk_loop_stop:
  brz               $m2,              __init_mem_int32_exit
  shr               $m3,              $m2,                INT32_PROC_KERNEL_BITS   //kernel_cnt
  shl               $m1,              $m3,                INT32_PROC_KERNEL_BITS   //shrink_proc_len
.align 8
  {
    rpt             $m3,              ((__init_mem_int32_internal_2nd_loop_end - __init_mem_int32_internal_2nd_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_int32_internal_2nd_loop_start:
  INIT_MEM_KERNEL_PROC
__init_mem_int32_internal_2nd_loop_end:
  sub               $m2,            $m2,               $m1
  brz               $m2,            __init_mem_int32_exit
.align 8
  {
    rpt             $m2,              ((__init_mem_int32_tail_loop_end - __init_mem_int32_tail_loop_start) / 8) - 1;
	  fnop
  }
__init_mem_int32_tail_loop_start:
  {
    st32step        $a0,              $m15,              $m0+=,             1
    fnop
  }
__init_mem_int32_tail_loop_end:
__init_mem_int32_exit:
  br                $lr
.Z15InitMemoryInt32Piii_stop:
	.size	_Z15InitMemoryInt32Piii, .Z15InitMemoryInt32Piii_stop-_Z15InitMemoryInt32Piii
	.section	.stack_sizes,"o",@progbits,.text._Z15InitMemoryInt32Piii
	.long	.Z15InitMemoryInt32Piii_start
	.byte	0
	.section	.text._Z15InitMemoryInt32Piii,"ax",@progbits
                                        # -- End function
	.ident	"clang version 13.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 6eeafc711ed2cca929acfbe560c20764774f9a28)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
#endif   //__INIT_MEMORY_ASM_COMMON__
#endif   //__IPU__