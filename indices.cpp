#include "alpaka/alpaka.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>

namespace a = alpaka;

using Idx = uint32_t;
using Elem = int;

int main() {
    const int xExtent = 4;
    const int yExtent = 5;

    a::PlatformCpu platformHost;
    a::DevCpu devHost = a::getDevByIdx(platformHost, 0);
    a::BufCpu<Elem, a::DimInt<2>, Idx> bufHost =
        a::allocBuf<Elem, Idx>(devHost, a::Vec<a::DimInt<2>, Idx>(yExtent, xExtent));
    assert(a::getExtents(bufHost).x() == a::getExtents(bufHost)[1]);
    assert(a::getExtents(bufHost).y() == a::getExtents(bufHost)[0]);
    assert(a::getExtents(bufHost).x() == xExtent);
    assert(a::getExtents(bufHost).y() == yExtent);

    std::ranges::generate(
        std::span(bufHost.data(), a::getExtents(bufHost).x() * a::getExtents(bufHost).y()),
        [i = 0]() mutable { return i--; });
    Elem* const bufHostData = bufHost.data();
    assert(*bufHostData == 0);
    assert(*(bufHostData + 2) == -2);
    assert(*(bufHostData + xExtent * yExtent - 1) == (-xExtent * yExtent + 1));
    assert((bufHost[a::Vec<a::DimInt<2>, Idx>(0, 0)] == *bufHostData));
    assert((bufHost[a::Vec<a::DimInt<2>, Idx>(0, 2)] == *(bufHostData + 2)));
    assert((bufHost[a::Vec<a::DimInt<2>, Idx>(yExtent - 1, xExtent - 1)] ==
            *(bufHostData + xExtent * yExtent - 1)));

    a::Vec<a::DimInt<2>, Idx> pitchesBytes = a::getPitchesInBytes(bufHost);
    assert(pitchesBytes.x() == pitchesBytes[1]);
    assert(pitchesBytes.y() == pitchesBytes[0]);
    assert(pitchesBytes.x() == sizeof(Elem));
    assert(pitchesBytes.y() == xExtent * sizeof(Elem));
}
