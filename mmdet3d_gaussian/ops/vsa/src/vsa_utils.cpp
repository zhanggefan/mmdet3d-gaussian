#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ball_query.h"
#include "group_points.h"
#include "sampling.h"
#include "voxel_query_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query, "ball_query_for_concatenate_input");
  m.def("voxel_query_wrapper", &voxel_query_wrapper_stack,
        "voxel_query_for_concatenate_input");

  m.def("furthest_point_sampling", &furthest_point_sampling,
        "furthest_point_sampling");

  m.def("group_points", &group_points, "group_points_for_concatenate_input");
  m.def("group_points_grad", &group_points_grad,
        "group_points_grad_for_concatenate_input");
}
