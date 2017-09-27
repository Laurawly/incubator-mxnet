/*!
 *  Copyright (c) 2017 by NVIDIA Corporation
 * \file nvtx.h
 * \brief Add support for NVTX
 */
#ifndef MXNET_COMMON_NVTX_H_
#define MXNET_COMMON_NVTX_H_

#if MXNET_USE_NVTX
#include <cuda.h>
#include <nvToolsExtCuda.h>

class nvtx {
 public:
  static const uint32_t kRed     = 0xFF0000;
  static const uint32_t kGreen   = 0x00FF00;
  static const uint32_t kBlue    = 0x0000FF;
  static const uint32_t kYellow  = 0xB58900;
  static const uint32_t kOrange  = 0xCB4B16;
  static const uint32_t kRed1    = 0xDC322F;
  static const uint32_t kMagenta = 0xD33682;
  static const uint32_t kViolet  = 0x6C71C4;
  static const uint32_t kBlue1   = 0x268BD2;
  static const uint32_t kCyan    = 0x2AA198;
  static const uint32_t kGreen1  = 0x859900;


  static void gpuRangeStart(const uint32_t rgb, const char *range_name) {
    nvtxEventAttributes_t att;
    att.version = NVTX_VERSION;
    att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    att.colorType = NVTX_COLOR_ARGB;
    att.color = rgb | 0xff000000;
    att.messageType = NVTX_MESSAGE_TYPE_ASCII;
    att.message.ascii = range_name;
    nvtxRangePushEx(&att);
  }

  static void gpuRangeStop() {
    nvtxRangePop();
  }
};

class CudaEventTimer {
 public:
  CudaEventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() {
    cudaEventRecord(start_);
  }

  float stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_, stop_;
};

#endif  // MXNET_USE_NVTX
#endif  // MXNET_COMMON_NVTX_H_
