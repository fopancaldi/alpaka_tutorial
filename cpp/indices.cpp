#include "alpaka_tutorial.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>

namespace a = alpaka;
namespace at = alpaka_tutorial;

template <typename T>
std::size_t SizeCast(T t) {
    return static_cast<std::size_t>(t);
}

struct CheckKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc&, const at::Elem* data, at::Vec2<at::Idx> extents,
                                  at::Vec2<at::Idx> pitches) const {
        const int xExt = static_cast<int>(extents.x());
        const int yExt = static_cast<int>(extents.y());
        for (int yIdx = 0; yIdx < yExt; ++yIdx) {
            for (int xIdx = 0; xIdx < xExt; ++xIdx) {
                ALPAKA_ASSERT(
                    data[SizeCast(xIdx + static_cast<int>(pitches.y() / sizeof(at::Elem)) *
                                             yIdx)] == -xIdx - yIdx * xExt);
            }
        }
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        for (int i = 0; i < static_cast<int>(extents.x() * extents.y()); ++i) {
            ALPAKA_ASSERT(data[SizeCast(i)] == -i);
        }
#endif
    }
};

int main() {
    using namespace alpaka_tutorial;

    constexpr int xExtent = 4;
    constexpr int yExtent = 7;

    PlatformH platfHost;
    assert(a::getDevCount(platfHost) > 0);
    DevH devHost = getDevByIdx(platfHost, 0);
    Platform platform;
    assert(a::getDevCount(platform) > 0);
    Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue(device);

    Buf2H<Elem> bufHost =
        a::allocBuf<Elem, Idx>(devHost, a::Vec<a::DimInt<2>, Idx>(yExtent, xExtent));
    assert(a::getExtents(bufHost).x() == xExtent);
    assert(a::getExtents(bufHost).y() == yExtent);
    assert(a::getExtents(bufHost).x() == a::getExtents(bufHost)[1]);
    assert(a::getExtents(bufHost).y() == a::getExtents(bufHost)[0]);

    Vec2<Idx> pitchesBytes = a::getPitchesInBytes(bufHost);
    assert(pitchesBytes.x() == pitchesBytes[1]);
    assert(pitchesBytes.y() == pitchesBytes[0]);
    assert(pitchesBytes.x() == sizeof(Elem));
    assert(pitchesBytes.y() == xExtent * sizeof(Elem));

    std::ranges::generate(
        std::span(bufHost.data(), a::getExtents(bufHost).x() * a::getExtents(bufHost).y()),
        [i = 0]() mutable { return i--; });
    const Elem* const bufHostData = bufHost.data();
    assert(*bufHostData == 0);
    assert(*(bufHostData + 2) == -2);
    assert(*(bufHostData + xExtent * yExtent - 1) == (-xExtent * yExtent + 1));
    assert(bufHost[Vec2<Idx>(0, 0)] == *bufHostData);
    assert(bufHost[Vec2<Idx>(0, 2)] == *(bufHostData + 2));
    assert(bufHost[Vec2<Idx>(3, 0)] == *(bufHostData + 3 * xExtent));
    assert(bufHost[Vec2<Idx>(yExtent - 1, xExtent - 1)] == *(bufHostData + xExtent * yExtent - 1));

    Buf2<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufHost));
    a::memcpy(queue, buf, bufHost);
    const Vec2<Idx> bufPitches = a::getPitchesInBytes(buf);
    assert(bufPitches.x() == sizeof(Elem));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    assert(bufPitches.y() == bufPitches.x() * sizeof(Elem));
#endif

    a::WorkDivMembers<Dim1, Idx> workDiv(Idx{1}, Idx{1}, Idx{1});
    a::exec<Acc1>(queue, workDiv, CheckKernel{}, buf.data(), a::getExtents(buf),
                  a::getPitchesInBytes(buf));
}
