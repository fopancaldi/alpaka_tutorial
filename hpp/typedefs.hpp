#pragma once

#include "basic_typedefs.hpp"
#include "config.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka_tutorial {

using Dim0D = alpaka::DimInt<0>;
using Dim1D = alpaka::DimInt<1>;
using Dim2D = alpaka::DimInt<2>;

template <typename TDim>
using Vec = alpaka::Vec<TDim, Idx>;
using Scalar = Vec<Dim0D>;
using Vec1D = Vec<Dim1D>;
using Vec2D = Vec<Dim2D>;

template <typename TElem, typename TDim>
using BufH = alpaka::Buf<DevHost, TElem, TDim, Idx>;
template <typename TElem>
using BufH0D = BufH<TElem, Dim0D>;
template <typename TElem>
using BufH1D = BufH<TElem, Dim1D>;
template <typename TElem>
using BufH2D = BufH<TElem, Dim2D>;

template <typename TElem, typename TDim>
using ViewH = alpaka::ViewPlainPtr<DevHost, TElem, TDim, Idx>;
template <typename TElem>
using ViewH0D = ViewH<TElem, Dim0D>;
template <typename TElem>
using ViewH1D = ViewH<TElem, Dim1D>;
template <typename TElem>
using ViewH2D = ViewH<TElem, Dim2D>;

template <typename TElem, typename TDim>
using Buf = alpaka::Buf<Device, TElem, TDim, Idx>;
template <typename TElem>
using Buf0D = Buf<TElem, Dim0D>;
template <typename TElem>
using Buf1D = Buf<TElem, Dim1D>;
template <typename TElem>
using Buf2D = Buf<TElem, Dim2D>;

template <typename TElem, typename TDim>
using View = alpaka::ViewPlainPtr<Device, TElem, TDim, Idx>;
template <typename TElem>
using View0D = View<TElem, Dim0D>;
template <typename TElem>
using View1D = View<TElem, Dim1D>;
template <typename TElem>
using View2D = View<TElem, Dim2D>;

template <typename TDim>
using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
using WorkDiv1D = WorkDiv<Dim1D>;
using WorkDiv2D = WorkDiv<Dim2D>;

using Acc1D = Acc<Dim1D>;
using Acc2D = Acc<Dim2D>;

} // namespace alpaka_tutorial
