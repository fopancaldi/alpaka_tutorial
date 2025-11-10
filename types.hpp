#pragma once

#include "alpaka/alpaka.hpp"
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace alpakaTutorial {

namespace a = alpaka;

using Elem = int;

using Dim1 = a::DimInt<1>;
using Dim2 = a::DimInt<2>;
using Idx = uint32_t;

using PlatformH = a::PlatformCpu;
using DevH = a::DevCpu;
using QueueH = a::Queue<DevH, a::Blocking>;

template <typename TElem, typename TDim>
using BufH = a::Buf<DevH, TElem, TDim, Idx>;
template <typename TElem>
using Buf1H = BufH<TElem, Dim1>;

template <typename TElem, typename TDim>
using ViewH = a::ViewPlainPtr<DevH, TElem, TDim, Idx>;
template <typename TElem>
using View1H = ViewH<TElem, Dim1>;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
using Platform = a::PlatformCpu;
using Device = a::DevCpu;
using Queue = a::Queue<Device, a::Blocking>;
template <typename TDim>
using Acc = alpaka::AccCpuSerial<TDim, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
using Platform = a::PlatformCpu;
using Device = a::DevCpu;
using Queue = a::Queue<Device, a::Blocking>;
template <typename TDim>
using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;

#else
#error "Define one backend configuration"
#endif

template <typename TElem, typename TDim>
using Buf = a::Buf<Device, TElem, TDim, Idx>;
template <typename TElem>
using Buf1 = Buf<TElem, Dim1>;

using Acc1 = Acc<Dim1>;

} // namespace alpakaTutorial
