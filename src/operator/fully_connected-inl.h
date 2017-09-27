/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./elemwise_op_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool cublas_algo_verbose;
  // This flag determines whether the new cublas_fully_connected object is invoked
  // (with detailed control over compute precision and algo), or the more-restrictive
  // mshadow use of cublas.  This flag parallels the control offered by the cudnn_off
  // flag seen with convolution.
  bool cublas_off;
  dmlc::optional<bool> cublas_tensor_core;
  dmlc::optional<int> cublas_algo_fwd;
  dmlc::optional<int> cublas_algo_bwd_data;
  dmlc::optional<int> cublas_algo_bwd_weights;
  int32_t cublas_algo_fwd_prec;
  int32_t cublas_algo_bwd_prec;

  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cublas_algo_verbose).set_default(false)
    .describe("Verboseness of algo selection. true = output selection, false = no output");
    DMLC_DECLARE_FIELD(cublas_off).set_default(false)
    .describe("Turn off full-control cublas for this layer.");
    DMLC_DECLARE_FIELD(cublas_tensor_core)
    .set_default(dmlc::optional<bool>())
    .describe("Allow Tensor Core math for default-chosen algos.");
    DMLC_DECLARE_FIELD(cublas_algo_fwd)
    .set_default(dmlc::optional<int>())
    .describe("Specified Forward GEMM Algorithm.");
    DMLC_DECLARE_FIELD(cublas_algo_bwd_data)
    .set_default(dmlc::optional<int>())
    .describe("Specified Backprop-to-Data GEMM Algorithm.");
    DMLC_DECLARE_FIELD(cublas_algo_bwd_weights)
    .set_default(dmlc::optional<int>())
    .describe("Specified Backprop-to-Weights GEMM Algorithm.");
    DMLC_DECLARE_FIELD(cublas_algo_fwd_prec)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("Precision of the computation of the forward GEMM kernel.\n    "
              "Default is determined by the tensor data type and the\n    "
              "MSHADOW_USE_PASCAL compiled-in flag.");
    DMLC_DECLARE_FIELD(cublas_algo_bwd_prec)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("Precision of the computation of the back-prop kernels.\n    "
              "Default is determined by the tensor data type and the\n    "
              "MSHADOW_USE_PASCAL compiled-in flag.");
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    out = dot(data, wmat.T());
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
      out += repmat(bias, data.size(0));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_grad[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);

#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<xpu, 2, DType>(s);
    Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }
    // gradient of data
    Tensor<xpu, 2, DType> gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Assign(gdata, req[fullc::kData], dot(grad, wmat));
  }

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    }
    CHECK_EQ(out_shape->size(), 1U);
    TShape dshape = (*in_shape)[fullc::kData];
    TShape oshape = (*out_shape)[0];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param_.num_hidden));
    }

    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param_.num_hidden));
    if (oshape.ndim() != 0) {
      dshape[0] = oshape[0];
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kData, dshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    nnvm::NodeAttrs attrs;
    attrs.name = "FullyConnected";
    return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_type, out_type, -1);
  }

  OperatorProperty* Copy() const override {
    FullyConnectedProp* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullc::kOut], in_data[fullc::kData], in_data[fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullc::kData], in_grad[fullc::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
