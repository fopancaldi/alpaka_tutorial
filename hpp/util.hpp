#pragma once

#include <concepts>

namespace alpaka_tutorial::util {

auto RatioRoundedUp(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

} // namespace alpaka_tutorial::util
