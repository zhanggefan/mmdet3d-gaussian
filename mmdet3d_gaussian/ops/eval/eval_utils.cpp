#include "torch/extension.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace eval {
namespace matcher {
py::array_t<int32_t> match_coco(const py::array_t<float> &cost_mat_,
                                const py::array_t<float> &cost_thrs_,
                                const py::array_t<bool> &is_ignore_,
                                const py::array_t<bool> &is_crowd_);
}
namespace affinity {
py::array_t<float> iou_3d(const py::array_t<float> &det_,
                          const py::array_t<float> &gt_, const float z_offset);
py::array_t<float> iou_bev(const py::array_t<float> &det_,
                           const py::array_t<float> &gt_);
py::array_t<float> trans_bev(const py::array_t<float> &det_,
                             const py::array_t<float> &gt_);
} // namespace affinity

} // namespace eval

using namespace eval;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("match_coco", &matcher::match_coco, py::arg("cost_mat_").noconvert(),
        py::arg("cost_thrs_").noconvert(), py::arg("is_ignore_").noconvert(),
        py::arg("is_crowd_").noconvert())
      .def("iou_3d", &affinity::iou_3d, py::arg("det_").noconvert(),
           py::arg("gt_").noconvert(), py::arg("z_offset") = 0.5f)
      .def("iou_bev", &affinity::iou_bev, py::arg("det_").noconvert(),
           py::arg("gt_").noconvert())
      .def("trans_bev", &affinity::trans_bev, py::arg("det_").noconvert(),
           py::arg("gt_").noconvert());
}