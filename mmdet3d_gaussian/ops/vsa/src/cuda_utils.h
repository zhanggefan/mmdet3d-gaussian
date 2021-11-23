#ifndef _STACK_CUDA_UTILS_H
#define _STACK_CUDA_UTILS_H

#include <cmath>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    if (!x.device().is_cuda()) {                                               \
      fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__,       \
              __LINE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CHECK_CONTIGUOUS(x)                                                    \
  do {                                                                         \
    if (!x.is_contiguous()) {                                                  \
      fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, \
              __LINE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#endif
