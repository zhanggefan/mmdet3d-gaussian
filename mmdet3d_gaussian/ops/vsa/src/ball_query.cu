/*
Stacked-batch-data version of ball query, modified from the original
implementation of official PointNet++ codes. Written by Shaoshuai Shi All Rights
Reserved 2019-2020.
*/
#include "cuda_utils.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

__global__ void ball_query_kernel(int B, int M, float radius, int nsample,
                                  const float *new_xyz,
                                  const int *new_xyz_batch_cnt,
                                  const float *xyz, const int *xyz_batch_cnt,
                                  int *idx) {
  // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
  // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
  // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
  // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
  // output:
  //      idx: (M, nsample)
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt_idx >= M)
    return;

  int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
  for (int k = 1; k < B; k++) {
    if (pt_idx < pt_cnt)
      break;
    pt_cnt += new_xyz_batch_cnt[k];
    bs_idx = k;
  }

  int xyz_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++)
    xyz_batch_start_idx += xyz_batch_cnt[k];
  // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx +=
  // new_xyz_batch_cnt[k];

  new_xyz += pt_idx * 3;
  xyz += xyz_batch_start_idx * 3;
  idx += pt_idx * nsample;

  float radius2 = radius * radius;
  float new_x = new_xyz[0];
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];
  int n = xyz_batch_cnt[bs_idx];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);
    if (d2 < radius2) {
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample)
        break;
    }
  }
  if (cnt == 0)
    idx[0] = -1;
}

int ball_query(float radius, int nsample, const at::Tensor &new_xyz_tensor,
               const at::Tensor &new_xyz_batch_cnt_tensor,
               const at::Tensor &xyz_tensor,
               const at::Tensor &xyz_batch_cnt_tensor, at::Tensor &idx_tensor) {
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT(new_xyz_batch_cnt_tensor);
  CHECK_INPUT(xyz_batch_cnt_tensor);
  const float *new_xyz = new_xyz_tensor.data_ptr<float>();
  const float *xyz = xyz_tensor.data_ptr<float>();
  const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
  const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
  int *idx = idx_tensor.data<int>();

  cudaError_t err;

  int B = xyz_batch_cnt_tensor.size(0);
  int M = new_xyz_tensor.size(0);

  dim3 blocks(DIVUP(M, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt,
      idx);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return 1;
}
