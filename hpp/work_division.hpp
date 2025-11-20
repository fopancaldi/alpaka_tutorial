#pragma once

#include "config.hpp"
#include "constants.hpp"
#include "typedefs.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>

// TODO: Replace alpaka::isAccelerator with alpaka::concepts::Acc

namespace alpaka_tutorial {

template <typename TAcc>
    requires a::isAccelerator<TAcc>
struct requires_single_elem_per_thread {};

template <typename TAcc>
    requires a::isAccelerator<TAcc> && (a::Dim<TAcc>::value == 1)
constexpr bool requires_single_elem_per_thread_v = requires_single_elem_per_thread<TAcc>::value;

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
template <typename TDim>
struct requires_single_elem_per_thread<Acc<TDim>> : public std::false_type {};

#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
template <typename TDim>
struct requires_single_elem_per_thread<Acc<TDim>> : public std::true_type {};
#endif

template <typename TAcc>
    requires a::isAccelerator<TAcc>
a::WorkDivMembers<Dim1, Idx> MakeWorkDiv(Idx gridBlocks, Idx blockElements) {
    if constexpr (requires_single_elem_per_thread_v<TAcc>) {
        return a::WorkDivMembers(Vec1<Idx>(gridBlocks), Vec1<Idx>(blockElements), Vec1<Idx>(1));
    } else {
        return a::WorkDivMembers(Vec1<Idx>(gridBlocks), Vec1<Idx>(1), Vec1<Idx>(blockElements));
    }
}

namespace internal {

auto RoundUpRatio(std::integral auto num, std::integral auto den) { return (num + den - 1) / den; }

} // namespace internal

template <typename TAcc>
    requires a::isAccelerator<TAcc>
a::WorkDivMembers<Dim1, Idx> MakeWorkDiv(Idx elements) {
    return MakeWorkDiv<TAcc>(internal::RoundUpRatio(elements, constants::blockSize), elements);
}

} // namespace alpaka_tutorial
