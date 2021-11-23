#pragma once
#include <torch/extension.h>

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

namespace voxelization {

#ifdef WITH_CUDA
at::Tensor dynamic_point_to_voxel_scatter_reduce_gpu(
    const at::Tensor &feats, const at::Tensor &coors_map,
    const at::Tensor &reduce_count, const reduce_t reduce_type);

std::vector<torch::Tensor>
dynamic_point_to_voxel_scatter_index_gpu(const torch::Tensor &coors);

void dynamic_point_to_voxel_backward_gpu(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type);
#endif

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else
    TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::SUM;
}

inline torch::Tensor dynamic_point_to_voxel_scatter_reduce(
    const torch::Tensor &feats, const torch::Tensor &coors_map,
    const at::Tensor &reduce_count, const std::string &reduce_type) {
  if (feats.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_point_to_voxel_scatter_reduce_gpu(
        feats, coors_map, reduce_count, convert_reduce_type(reduce_type));
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "do not support cpu yet");
  return torch::Tensor();
}

inline std::vector<torch::Tensor>
dynamic_point_to_voxel_scatter_index(const torch::Tensor &coors) {
  if (coors.device().is_cuda()) {
#ifdef WITH_CUDA
    return dynamic_point_to_voxel_scatter_index_gpu(coors);
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "do not support cpu yet");
  return std::vector<torch::Tensor>();
}

inline void dynamic_point_to_voxel_backward(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const std::string &reduce_type) {
  if (grad_feats.device().is_cuda()) {
#ifdef WITH_CUDA
    dynamic_point_to_voxel_backward_gpu(grad_feats, grad_reduced_feats, feats,
                                        reduced_feats, coors_idx, reduce_count,
                                        convert_reduce_type(reduce_type));
    return;
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "do not support cpu yet");
}

} // namespace voxelization
