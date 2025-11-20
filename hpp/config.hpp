#pragma once

#include "alpaka/alpaka.hpp"
#include "basic_typedefs.hpp"

namespace alpaka_tutorial {

namespace a = alpaka;

using PlatformH = a::PlatformCpu;
using DevH = a::DevCpu;
using QueueH = a::Queue<DevH, a::Blocking>;

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
using Platform = a::PlatformCpu;
using Device = a::DevCpu;
using Queue = a::Queue<Device, a::Blocking>;
template <typename TDim>
using Acc = a::AccCpuSerial<TDim, Idx>;

#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
using Platform = a::PlatformCpu;
using Device = a::DevCpu;
using Queue = a::Queue<Device, a::Blocking>;
template <typename TDim>
using Acc = a::AccCpuTbbBlocks<TDim, Idx>;

#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
using Platform = a::PlatformCudaRt;
using Device = a::DevCudaRt;
using Queue = a::QueueCudaRtNonBlocking;
template <typename TDim>
using Acc = a::AccGpuCudaRt<TDim, Idx>;

#else
#error "Define one backend configuration"
#endif

} // namespace alpaka_tutorial
