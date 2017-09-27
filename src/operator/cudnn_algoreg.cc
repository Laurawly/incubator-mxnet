/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_algoreg.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./cudnn_algoreg-inl.h"
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include <sstream>
#include <unordered_map>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
CuDNNAlgoReg *CuDNNAlgoReg::Get() {
  static CuDNNAlgoReg *ptr = new CuDNNAlgoReg();
  return ptr;
}

// Program start-up check that the version of cudnn compiled
// against matches the linked-against version.
bool CuDNNVersionCheck() {
  size_t linkedAgainstCudnnVersion = cudnnGetVersion();
  if (linkedAgainstCudnnVersion != CUDNN_VERSION)
    LOG(FATAL) << "cuDNN library mismatch: linked-against version " <<
               linkedAgainstCudnnVersion << " != compiled-against version " <<
               CUDNN_VERSION;
  return true;
}

// Global init will fail if runtime and compile-time versions mismatch.
bool cudnn_version_matches = CuDNNVersionCheck();
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet
