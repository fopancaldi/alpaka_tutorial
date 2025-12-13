#pragma once

#include "basic_typedefs.hpp"
#include "concepts.hpp"

namespace alpaka_tutorial::constants {

Idx constexpr bufLength = 100;

namespace detail {

Idx constexpr totalBlockElements = 16;

template <noalpaka::concepts::Acc TAcc>
struct BlockElements {};

template <concepts::Acc1D TAcc>
struct BlockElements<TAcc> {
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    static auto constexpr value = alpaka::Vec<Dim, Idx>(totalBlockElements);
};

template <concepts::Acc2D TAcc>
struct BlockElements<TAcc> {
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    static auto constexpr value = alpaka::Vec<Dim, Idx>(4, 4);
    static_assert(value.prod() == totalBlockElements);
};

} // namespace detail

template <noalpaka::concepts::Acc TAcc>
alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> constexpr blockElements =
    detail::BlockElements<TAcc>::value;

} // namespace alpaka_tutorial::constants
