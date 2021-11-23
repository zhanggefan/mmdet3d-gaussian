/*
Stacked-batch-data version of point grouping, modified from the original
implementation of official PointNet++ codes. Written by Shaoshuai Shi All Rights
Reserved 2019-2020.
*/

#ifndef _STACK_GROUP_POINTS_H
#define _STACK_GROUP_POINTS_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>
#include <vector>

int group_points(const at::Tensor &features_tensor,
                 const at::Tensor &features_batch_cnt_tensor,
                 const at::Tensor &idx_tensor,
                 const at::Tensor &idx_batch_cnt_tensor,
                 at::Tensor &out_tensor);

int group_points_grad(const at::Tensor &grad_out_tensor,
                      const at::Tensor &idx_tensor,
                      const at::Tensor &idx_batch_cnt_tensor,
                      const at::Tensor &features_batch_cnt_tensor,
                      at::Tensor &grad_features_tensor);

#endif
