/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
#include "./cublas_fully_connected-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_CUDA && CUDA_VERSION >= 8000
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cublas_off || !CuBLASFullyConnectedOp<DType>::Supports(param, ctx))
      op = new FullyConnectedOp<gpu, DType>(param);
    else
      op = new CuBLASFullyConnectedOp<DType>(param, ctx);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new FullyConnectedOp<gpu, DType>(param);
  })
#endif
  return op;
}
}  // namespace op
}  // namespace mxnet
