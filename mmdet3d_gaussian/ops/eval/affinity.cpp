#include "rbox_utils.hpp"
#include <float.h>
#include <pybind11/numpy.h>

namespace eval {
namespace affinity {
namespace py = pybind11;
py::array_t<float> iou_3d(const py::array_t<float> &det_,
                          const py::array_t<float> &gt_, const float z_offset) {
  auto det = det_.unchecked<2>();
  auto gt = gt_.unchecked<2>();

  auto num_det = det.shape(0), num_gt = gt.shape(0);

  auto iou3d_mat_ = py::array_t<float>({num_det, num_gt});
  auto iou3d_mat = iou3d_mat_.mutable_unchecked<2>();

  for (int di = 0; di < num_det; di++) {
    for (int gi = 0; gi < num_gt; gi++) {
      using namespace rbox;
      auto *d_ptr = det.data(di, 0);
      auto *g_ptr = gt.data(gi, 0);
      RotatedBox<float> d{d_ptr[0], d_ptr[1], d_ptr[3], d_ptr[4], d_ptr[6]};
      RotatedBox<float> g{g_ptr[0], g_ptr[1], g_ptr[3], g_ptr[4], g_ptr[6]};
      auto bev_inter = rotated_boxes_intersection(d, g);
      auto d_z_bottom = d_ptr[2] + (z_offset - 0.5f) * d_ptr[5];
      auto g_z_bottom = g_ptr[2] + (z_offset - 0.5f) * g_ptr[5];
      auto d_z_top = d_ptr[2] + (z_offset + 0.5f) * d_ptr[5];
      auto g_z_top = g_ptr[2] + (z_offset + 0.5f) * g_ptr[5];

      auto z_bottom = d_z_bottom > g_z_bottom ? d_z_bottom : g_z_bottom;
      auto z_top = d_z_top < g_z_top ? d_z_top : g_z_top;
      auto z_inter = z_top - z_bottom;
      z_inter = z_inter < 0.f ? 0.f : z_inter;

      auto d_vol = d_ptr[3] * d_ptr[4] * d_ptr[5];
      auto g_vol = g_ptr[3] * g_ptr[4] * g_ptr[5];
      auto inter_vol = bev_inter * z_inter;
      inter_vol = inter_vol < 0.f ? 0.f : inter_vol;
      inter_vol = inter_vol > d_vol ? d_vol : inter_vol;
      inter_vol = inter_vol > g_vol ? g_vol : inter_vol;
      auto union_vol = d_vol + g_vol - inter_vol;
      union_vol = union_vol < FLT_EPSILON ? FLT_EPSILON : union_vol;

      iou3d_mat(di, gi) = inter_vol / union_vol;
    }
  }
  return iou3d_mat_;
}

py::array_t<float> iou_bev(const py::array_t<float> &det_,
                           const py::array_t<float> &gt_) {
  auto det = det_.unchecked<2>();
  auto gt = gt_.unchecked<2>();

  auto num_det = det.shape(0), num_gt = gt.shape(0);

  auto ioubev_mat_ = py::array_t<float>({num_det, num_gt});
  auto ioubev_mat = ioubev_mat_.mutable_unchecked<2>();

  for (int di = 0; di < num_det; di++) {
    for (int gi = 0; gi < num_gt; gi++) {
      using namespace rbox;
      auto *d_ptr = det.data(di, 0);
      auto *g_ptr = gt.data(gi, 0);
      RotatedBox<float> d{d_ptr[0], d_ptr[1], d_ptr[3], d_ptr[4], d_ptr[6]};
      RotatedBox<float> g{g_ptr[0], g_ptr[1], g_ptr[3], g_ptr[4], g_ptr[6]};
      auto d_area = d_ptr[3] * d_ptr[4];
      auto g_area = g_ptr[3] * g_ptr[4];
      auto inter_area = rotated_boxes_intersection(d, g);
      inter_area = inter_area < 0.f ? 0.f : inter_area;
      inter_area = inter_area > d_area ? d_area : inter_area;
      inter_area = inter_area > g_area ? g_area : inter_area;

      auto union_area = d_area + g_area - inter_area;
      union_area = union_area < FLT_EPSILON ? FLT_EPSILON : union_area;
      ioubev_mat(di, gi) = inter_area / union_area;
    }
  }
  return ioubev_mat_;
}

py::array_t<float> trans_bev(const py::array_t<float> &det_,
                             const py::array_t<float> &gt_) {
  auto det = det_.unchecked<2>();
  auto gt = gt_.unchecked<2>();

  auto num_det = det.shape(0), num_gt = gt.shape(0);

  auto trans_mat_ = py::array_t<float>({num_det, num_gt});
  auto trans_mat = trans_mat_.mutable_unchecked<2>();
  
  for (int di = 0; di < num_det; di++) {
    for (int gi = 0; gi < num_gt; gi++) {
      auto *d_ptr = det.data(di, 0);
      auto *g_ptr = gt.data(gi, 0);
      auto d_cx = d_ptr[0], d_cy = d_ptr[1];
      auto g_cx = g_ptr[0], g_cy = g_ptr[1];
      auto dist =
          sqrt((d_cx - g_cx) * (d_cx - g_cx) + (d_cy - g_cy) * (d_cy - g_cy));
      trans_mat(di, gi) = dist;
    }
  }
  return trans_mat_;
}
} // namespace affinity
} // namespace eval