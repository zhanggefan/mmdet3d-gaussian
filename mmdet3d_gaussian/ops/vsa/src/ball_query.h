/*
Stacked-batch-data version of ball query, modified from the original
implementation of official PointNet++ codes. Written by Shaoshuai Shi All Rights
Reserved 2019-2020.
*/

#ifndef _STACK_BALL_QUERY_H
#define _STACK_BALL_QUERY_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

int ball_query(float radius, int nsample, const at::Tensor &new_xyz_tensor,
               const at::Tensor &new_xyz_batch_cnt_tensor,
               const at::Tensor &xyz_tensor,
               const at::Tensor &xyz_batch_cnt_tensor, at::Tensor &idx_tensor);

#endif
