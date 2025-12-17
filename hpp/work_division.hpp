#pragma once

#include "concepts.hpp"
#include "constants.hpp"
#include "typedefs.hpp"
#include "util.hpp"

#include <alpaka/alpaka.hpp>
#include <concepts>

namespace alpaka_tutorial {

template <noalpaka::concepts::Acc TAcc>
    requires std::same_as<alpaka::Idx<TAcc>, Idx> and (alpaka::Dim<TAcc>::value > 0)
alpaka::WorkDivMembers<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
MakeWorkDiv(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> blocks,
            alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> blockElements) {
    using Dim = alpaka::Dim<TAcc>;

    if constexpr (alpaka::isSingleThreadAcc<TAcc>) {
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
