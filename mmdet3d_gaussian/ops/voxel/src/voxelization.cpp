#include "voxelization.h"
#include <torch/extension.h>

namespace voxelization {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dynamic_point_to_voxel_scatter_reduce",
        &dynamic_point_to_voxel_scatter_reduce,
        "dynamic point to voxel scatter reduce");
  m.def("dynamic_point_to_voxel_scatter_index",
        &dynamic_point_to_voxel_scatter_index,
        "dynamic point to voxel scatter index");
  m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward,
        "dynamic point to voxel backward");
}

} // namespace voxelization
