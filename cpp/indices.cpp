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
    template <at::concepts::Acc1D TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&, at::Elem const* data, at::Vec2D extents,
                                  at::Vec2D pitches) const {
        for (at::Idx yIdx = 0; yIdx < extents.y(); ++yIdx) {
            for (at::Idx xIdx = 0; xIdx < extents.x(); ++xIdx) {
                ALPAKA_ASSERT(
                    data[xIdx + yIdx * pitches.y() / static_cast<at::Idx>(sizeof(at::Elem))] ==
                    -static_cast<int>(xIdx + yIdx * extents.x()));
            }
        }
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        for (at::Idx i = 0; i < extents.x() * extents.y(); ++i) {
            ALPAKA_ASSERT(data[i] == -static_cast<int>(i));
        }
#endif
    }
};

int main() {
    using namespace at;

    constexpr int xExtent = 4;
    constexpr int yExtent = 7;

    PlatformHost platfHost;
    assert(a::getDevCount(platfHost) > 0);
    DevHost devHost = getDevByIdx(platfHost, 0);
    Platform platform;
    assert(a::getDevCount(platform) > 0);
    Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue(device);

    BufH2D<Elem> bufHost = a::allocBuf<Elem, Idx>(devHost, Vec2D(yExtent, xExtent));
    assert(a::getExtents(bufHost).x() == xExtent);
    assert(a::getExtents(bufHost).y() == yExtent);
    assert(a::getExtents(bufHost).x() == a::getExtents(bufHost)[1]);
    assert(a::getExtents(bufHost).y() == a::getExtents(bufHost)[0]);

    Vec2D const pitchesBytes = a::getPitchesInBytes(bufHost);
    assert(pitchesBytes.x() == pitchesBytes[1]);
    assert(pitchesBytes.y() == pitchesBytes[0]);
    assert(pitchesBytes.x() == sizeof(Elem));
    assert(pitchesBytes.y() == xExtent * sizeof(Elem));

    std::ranges::generate(
        std::span(bufHost.data(), a::getExtents(bufHost).x() * a::getExtents(bufHost).y()),
        [i = 0]() mutable { return i--; });
    Elem const* const bufHostData = bufHost.data();
    assert(*bufHostData == 0);
    assert(*(bufHostData + 2) == -2);
    assert(*(bufHostData + xExtent * yExtent - 1) == (-xExtent * yExtent + 1));
    assert(bufHost[Vec2D(0, 0)] == *bufHostData);
    assert(bufHost[Vec2D(0, 2)] == *(bufHostData + 2));
    assert(bufHost[Vec2D(3, 0)] == *(bufHostData + 3 * xExtent));
    assert(bufHost[Vec2D(yExtent - 1, xExtent - 1)] == *(bufHostData + xExtent * yExtent - 1));

    Buf2D<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufHost));
    a::memcpy(queue, buf, bufHost);
    Vec2D const bufPitches = a::getPitchesInBytes(buf);
    assert(bufPitches.x() == sizeof(Elem));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    assert(bufPitches.y() == bufPitches.x() * sizeof(Elem));
#endif

    WorkDiv1D const workDiv(Idx{1}, Idx{1}, Idx{1});
    a::exec<Acc1D>(queue, workDiv, CheckKernel{}, buf.data(), a::getExtents(buf),
                   a::getPitchesInBytes(buf));
}
