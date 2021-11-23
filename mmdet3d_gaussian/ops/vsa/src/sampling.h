#ifndef _SAMPLING_H
#define _SAMPLING_H

#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>
#include <vector>

int furthest_point_sampling(const at::Tensor &points_tensor,
                            at::Tensor &temp_tensor, at::Tensor &idx_tensor);

#endif
