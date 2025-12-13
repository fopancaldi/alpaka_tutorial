#pragma once

#include "basic_typedefs.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka_tutorial {

using PlatformHost = alpaka::PlatformCpu;
using DevHost = alpaka::DevCpu;
using QueueHost = alpaka::Queue<DevHost, alpaka::Blocking>;

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
using Platform = alpaka::PlatformCpu;
using Device = alpaka::DevCpu;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;
template <typename TDim>
using Acc = alpaka::AccCpuSerial<TDim, Idx>;

#elif defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
using Platform = alpaka::PlatformCpu;
using Device = alpaka::DevCpu;
using Queue = alpaka::Queue<Device, alpaka::Blocking>;
template <typename TDim>
using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;

#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
using Platform = alpaka::PlatformCudaRt;
using Device = alpaka::DevCudaRt;
using Queue = alpaka::QueueCudaRtNonBlocking;
template <typename TDim>
using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;

#else
#error "Define one backend configuration"
#endif

} // namespace alpaka_tutorial
