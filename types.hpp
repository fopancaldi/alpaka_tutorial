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

using PlatformHost = a::PlatformCpu;
using DevHost = a::DevCpu;
using QueueHost = a::Queue<DevHost, a::Blocking>;

template <typename TElem, typename TDim>
using BufH = a::Buf<DevHost, TElem, TDim, Idx>;
template <typename TElem>
using Buf1H = a::Buf<DevHost, TElem, Dim1, Idx>;

template <typename TElem, typename TDim>
using ViewH = a::ViewPlainPtr<DevHost, TElem, TDim, Idx>;
template <typename TElem>
using View1H = a::ViewPlainPtr<DevHost, TElem, Dim1, Idx>;

// TODO: Add views for buffers?
template <typename TElem, typename TDim>
using ViewCH = a::ViewConst<std::decay_t<ViewH<TElem, TDim>>>;
template <typename TElem>
using ViewC1H = a::ViewConst<std::decay_t<View1H<TElem>>>;

using Platform = a::PlatformCpu;
using Device = a::DevCpu;
using Queue = a::Queue<Device, a::NonBlocking>;
template <typename TElem>
using Buf1 = a::Buf<Device, TElem, Dim1, Idx>;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
template <typename TDim>
using Acc = alpaka::AccCpuSerial<TDim, Idx>;
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
template <typename TDim>
using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
#else
#error "Define one backend configuration"
#endif

} // namespace alpakaTutorial
