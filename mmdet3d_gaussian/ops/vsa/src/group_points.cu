/*
Stacked-batch-data version of point grouping, modified from the original
implementation of official PointNet++ codes. Written by Shaoshuai Shi All Rights
Reserved 2019-2020.
*/

#include "cuda_utils.h"
#include "group_points.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void group_points_grad_kernel_stack(
    int B, int M, int C, int N, int nsample, const float *grad_out,
    const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt,
    float *grad_features) {
  // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the
  // output from forward :param idx: (M1 + M2 ..., nsample) tensor containing
  // the indicies of features to group with :param idx_batch_cnt: (batch_size)
  // [M1 + M2 ...] tensor containing the indicies of features to group with
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indicies of features to group with :return:
  //     grad_features: (N1 + N2 ..., C) gradient of the features
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample)
    return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++) {
    if (pt_idx < pt_cnt)
      break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++)
    features_batch_start_idx += features_batch_cnt[k];

  grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
  idx += pt_idx * nsample + sample_idx;
  grad_features += (features_batch_start_idx + idx[0]) * C + C_idx;

  atomicAdd(grad_features, grad_out[0]);
}

__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
                                          const float *features,
                                          const int *features_batch_cnt,
                                          const int *idx,
                                          const int *idx_batch_cnt,
                                          float *out) {
  // :param features: (N1 + N2 ..., C) tensor of features to group
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indicies of features to group with :param idx: (M1 + M2 ..., nsample)
  // tensor containing the indicies of features to group with :param
  // idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of
  // features to group with :return:
  //     output: (M1 + M2, C, nsample) tensor
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample)
    return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++) {
    if (pt_idx < pt_cnt)
      break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++)
    features_batch_start_idx += features_batch_cnt[k];
  features += features_batch_start_idx * C;

  idx += pt_idx * nsample + sample_idx;
  int in_idx = idx[0] * C + C_idx;
  int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

  out[out_idx] = features[in_idx];
}

int group_points_grad(const at::Tensor &grad_out_tensor,
                      const at::Tensor &idx_tensor,
                      const at::Tensor &idx_batch_cnt_tensor,
                      const at::Tensor &features_batch_cnt_tensor,
                      at::Tensor &grad_features_tensor) {

  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(grad_features_tensor);

  const float *grad_out = grad_out_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data_ptr<int>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data_ptr<int>();
  float *grad_features = grad_features_tensor.data_ptr<float>();

  cudaError_t err;

  int B = idx_batch_cnt_tensor.size(0);
  int N = grad_features_tensor.size(0), C = grad_features_tensor.size(1);
  int M = idx_tensor.size(0), nsample = idx_tensor.size(1);

  dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  group_points_grad_kernel_stack<<<blocks, threads, 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt,
      grad_features);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return 1;
}

int group_points(const at::Tensor &features_tensor,
                 const at::Tensor &features_batch_cnt_tensor,
                 const at::Tensor &idx_tensor,
                 const at::Tensor &idx_batch_cnt_tensor,
                 at::Tensor &out_tensor) {

  CHECK_INPUT(features_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);
  CHECK_INPUT(out_tensor);

  const float *features = features_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data_ptr<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data_ptr<int>();
  float *out = out_tensor.data_ptr<float>();

  cudaError_t err;

  int B = idx_batch_cnt_tensor.size(0);
  int C = features_tensor.size(1);
  int M = idx_tensor.size(0), nsample = idx_tensor.size(1);

  dim3 blocks(DIVUP(M * C * nsample,
                    THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_kernel_stack<<<blocks, threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return 1;
}