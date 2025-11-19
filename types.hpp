#pragma once

#include "alpaka/alpaka.hpp"
#include <cstdint>

namespace alpaka_tutorial {

namespace a = alpaka;

using Elem = int;

using Idx = uint32_t;
using Dim0 = a::DimInt<0>;
using Dim1 = a::DimInt<1>;
using Dim2 = a::DimInt<2>;

template <typename TElem>
using Vec0 = a::Vec<Dim0, TElem>;
template <typename TElem>
using Vec1 = a::Vec<Dim1, TElem>;
template <typename TElem>
using Vec2 = a::Vec<Dim2, TElem>;

using PlatformH = a::PlatformCpu;
using DevH = a::DevCpu;
using QueueH = a::Queue<DevH, a::Blocking>;

template <typename TElem, typename TDim>
using BufH = a::Buf<DevH, TElem, TDim, Idx>;
template <typename TElem>
using Buf0H = BufH<TElem, Dim0>;
template <typename TElem>
using Buf1H = BufH<TElem, Dim1>;
template <typename TElem>
using Buf2H = BufH<TElem, Dim2>;

template <typename TElem, typename TDim>
using ViewH = a::ViewPlainPtr<DevH, TElem, TDim, Idx>;
template <typename TElem>
using View0H = ViewH<TElem, Dim0>;
template <typename TElem>
using View1H = ViewH<TElem, Dim1>;
template <typename TElem>
using View2H = ViewH<TElem, Dim2>;

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

template <typename TElem, typename TDim>
using Buf = a::Buf<Device, TElem, TDim, Idx>;
template <typename TElem>
using Buf0 = Buf<TElem, Dim0>;
template <typename TElem>
using Buf1 = Buf<TElem, Dim1>;
template <typename TElem>
using Buf2 = Buf<TElem, Dim2>;

template <typename TElem, typename TDim>
using View = a::ViewPlainPtr<Device, TElem, TDim, Idx>;
template <typename TElem>
using View0 = View<TElem, Dim0>;
template <typename TElem>
using View1 = View<TElem, Dim1>;
template <typename TElem>
using View2 = View<TElem, Dim2>;

using Acc1 = Acc<Dim1>;

} // namespace alpaka_tutorial
