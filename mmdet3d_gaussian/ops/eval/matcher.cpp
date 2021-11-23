#include <cstring>
#include <pybind11/numpy.h>

namespace eval {
namespace matcher {
namespace py = pybind11;

py::array_t<int32_t> match_coco(const py::array_t<float> &cost_mat_,
                                const py::array_t<float> &cost_thrs_,
                                const py::array_t<bool> &is_ignore_,
                                const py::array_t<bool> &is_crowd_) {

  auto cost_mat = cost_mat_.unchecked<2>();
  auto cost_thrs = cost_thrs_.unchecked<1>();
  auto is_ignore = is_ignore_.unchecked<1>();
  auto is_crowd = is_crowd_.unchecked<1>();

  auto num_det = cost_mat.shape(0), num_gt = cost_mat.shape(1),
       num_cost_thr = cost_thrs.shape(0);

  auto matched_gt_ = py::array_t<int32_t>({num_cost_thr, num_det});
  auto matched_gt = matched_gt_.mutable_unchecked<2>();

  bool *gt_matched = new bool[num_cost_thr * num_gt];
  std::memset(gt_matched, 0, num_cost_thr * num_gt * sizeof(bool));

  for (int cost_thr_idx = 0; cost_thr_idx < num_cost_thr; cost_thr_idx++) {
    for (int det_idx = 0; det_idx < num_det; det_idx++) {
      float cost_thr = cost_thrs(cost_thr_idx);
      float cost = cost_thr;
      int matching_gt = -1;
      for (int gt_idx = 0; gt_idx < num_gt; gt_idx++) {
        bool ignore = is_ignore(gt_idx);
        bool crowd = is_crowd(gt_idx);
        if (gt_matched[cost_thr_idx * num_gt + gt_idx] && (!crowd))
          continue;
        // if dt matched to reg gt, and on ignore gt, continue
        float cost_mat_val = cost_mat(det_idx, gt_idx);
        if (matching_gt == -1) { // no match yet
          if (cost_mat_val <= cost) {
            cost = cost_mat_val;
            matching_gt = gt_idx;
          }
        } else {
          if (is_ignore(matching_gt)) { // matched to ignore gt.
            if ((!ignore)) { // this gt is non-ignore, match it as long as the
                             // the cost is lower than threshold.
              if (cost_mat_val <= cost_thr) {
                cost = cost_mat_val;
                matching_gt = gt_idx;
              }
            } else { // this gt is ignore, match it only when the cost is lower
                     // than current.
              if (cost_mat_val <= cost) {
                cost = cost_mat_val;
                matching_gt = gt_idx;
              }
            }
          } else { // matched to non-ignore gt.
            if ((!ignore) && (cost_mat_val <= cost)) {
              cost = cost_mat_val;
              matching_gt = gt_idx;
            }
          }
        }
      }
      if (matching_gt != -1) {
        gt_matched[cost_thr_idx * num_gt + matching_gt] = true;
      }
      matched_gt(cost_thr_idx, det_idx) = matching_gt;
    }
  }
  delete[] gt_matched;
  return matched_gt_;
}
} // namespace matcher
} // namespace eval
