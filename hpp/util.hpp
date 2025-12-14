#pragma once

#include "concepts.hpp"
#include "typedefs.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>
#include <type_traits>

namespace alpaka_tutorial {

namespace util {

auto RatioRoundedUp(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

} // namespace util

// TODO: Find a better name
template <nostd::pointer TPtr, noalpaka::concepts::Dim TDim>
ALPAKA_FN_HOST_ACC auto& PtrAt(TPtr const& ptr, Vec<TDim> idx, Vec<TDim> pitch) {
    using T = std::remove_pointer_t<TPtr>;

    static_assert(sizeof(T) <= std::numeric_limits<Idx>::max());
    return ptr[(idx * pitch).sum() / sizeof(T)];
}

} // namespace alpaka_tutorial
