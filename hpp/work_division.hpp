#pragma once

#include "concepts.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "typedefs.hpp"
#include "util.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>

namespace alpaka_tutorial {

template <noalpaka::concepts::Acc TAcc>
struct requires_single_thread_per_block;

template <noalpaka::concepts::Acc TAcc>
constexpr bool requires_single_thread_per_block_v = requires_single_thread_per_block<TAcc>::value;

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
template <typename TDim>
struct requires_single_thread_per_block<Acc<TDim>> : public std::true_type {};

#elif defined ALPAKA_ACC_GPU_CUDA_ENABLED
template <typename TDim>
struct requires_single_thread_per_block<Acc<TDim>> : public std::false_type {};
#endif

template <noalpaka::concepts::Acc TAcc>
    requires std::same_as<alpaka::Idx<TAcc>, Idx> and (alpaka::Dim<TAcc>::value > 0)
alpaka::WorkDivMembers<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
MakeWorkDiv(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> blocks,
            alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> blockElements) {
    using Dim = alpaka::Dim<TAcc>;

    if constexpr (requires_single_thread_per_block_v<TAcc>) {
        return alpaka::WorkDivMembers(blocks, Vec<Dim>::ones(), blockElements);
    } else {
        return alpaka::WorkDivMembers(blocks, blockElements, Vec<Dim>::ones());
    }
}

template <noalpaka::concepts::Acc TAcc>
    requires std::same_as<alpaka::Idx<TAcc>, Idx> and (alpaka::Dim<TAcc>::value > 0)
alpaka::WorkDivMembers<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
MakeWorkDiv(Vec<alpaka::Dim<TAcc>> elements) {
    using Dim = alpaka::Dim<TAcc>;

    Vec<Dim> blocks;
    std::ranges::transform(elements, constants::blockElements<TAcc>, blocks.begin(),
                           [](Idx es, Idx bes) { return util::RatioRoundedUp(es, bes); });
    return MakeWorkDiv<TAcc>(blocks, constants::blockElements<TAcc>);
}

} // namespace alpaka_tutorial
