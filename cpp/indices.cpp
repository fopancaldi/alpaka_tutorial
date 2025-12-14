#include "alpaka_tutorial.hpp"

#include <algorithm>
#include <cassert>
#include <span>

namespace a = alpaka;
namespace at = alpaka_tutorial;

struct CheckKernel {
    template <at::concepts::Acc1D TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&, at::Elem const* data, at::Vec2D extent,
                                  at::Vec2D pitch) const {
        for (at::Idx yIdx = 0; yIdx < extent.y(); ++yIdx) {
            for (at::Idx xIdx = 0; xIdx < extent.x(); ++xIdx) {
                ALPAKA_ASSERT(
                    data[xIdx + yIdx * pitch.y() / static_cast<at::Idx>(sizeof(at::Elem))] ==
                    -static_cast<int>(xIdx + yIdx * extent.x()));
            }
        }
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        for (at::Idx i = 0; i < extent.x() * extent.y(); ++i) {
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

    BufH2D<Elem> bufH = a::allocBuf<Elem, Idx>(devHost, Vec2D(yExtent, xExtent));
    assert(a::getExtents(bufH).x() == xExtent);
    assert(a::getExtents(bufH).y() == yExtent);
    assert(a::getExtents(bufH).x() == a::getExtents(bufH)[1]);
    assert(a::getExtents(bufH).y() == a::getExtents(bufH)[0]);

    Vec2D const pitchesBytes = a::getPitchesInBytes(bufH);
    assert(pitchesBytes.x() == pitchesBytes[1]);
    assert(pitchesBytes.y() == pitchesBytes[0]);
    assert(pitchesBytes.x() == sizeof(Elem));
    assert(pitchesBytes.y() == xExtent * sizeof(Elem));

    std::ranges::generate(std::span(bufH.data(), a::getExtents(bufH).x() * a::getExtents(bufH).y()),
                          [i = 0]() mutable { return i--; });
    Elem const* const bufHData = bufH.data();
    assert(*bufHData == 0);
    assert(*(bufHData + 2) == -2);
    assert(*(bufHData + xExtent * yExtent - 1) == (-xExtent * yExtent + 1));
    assert(bufH[Vec2D(0, 0)] == *bufHData);
    assert(bufH[Vec2D(0, 2)] == *(bufHData + 2));
    assert(bufH[Vec2D(3, 0)] == *(bufHData + 3 * xExtent));
    assert(bufH[Vec2D(yExtent - 1, xExtent - 1)] == *(bufHData + xExtent * yExtent - 1));

    Buf2D<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufH));
    a::memcpy(queue, buf, bufH);
    Vec2D const bufPitches = a::getPitchesInBytes(buf);
    assert(bufPitches.x() == sizeof(Elem));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    assert(bufPitches.y() == bufPitches.x() * sizeof(Elem));
#endif

    WorkDiv1D const workDiv(Idx{1}, Idx{1}, Idx{1});
    a::exec<Acc1D>(queue, workDiv, CheckKernel{}, buf.data(), a::getExtents(buf),
                   a::getPitchesInBytes(buf));
}
