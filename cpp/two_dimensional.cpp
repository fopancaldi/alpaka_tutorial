#include "alpaka_tutorial.hpp"

#include <algorithm>
#include <cassert>
#include <span>

namespace a = alpaka;
namespace at = alpaka_tutorial;

struct CheckKernel {
    template <at::concepts::Acc2D TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, at::Elem const* data, at::Vec2D extent,
                                  at::Vec2D pitch) const {
        for (at::Vec2D idx : a::uniformElementsND(acc, extent)) {
            ALPAKA_ASSERT(
                data[idx.x() + idx.y() * pitch.y() / static_cast<at::Idx>(sizeof(at::Elem))] ==
                -static_cast<at::Elem>(idx.x() + idx.y() * extent.x()));
            ALPAKA_ASSERT(at::PtrAt(data, idx, pitch) ==
                          -static_cast<at::Elem>(idx.x() + idx.y() * extent.x()));
        }
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        if (a::oncePerGrid(acc)) {
            for (at::Idx i = 0; i < extent.x() * extent.y(); ++i) {
                ALPAKA_ASSERT(data[i] == -static_cast<at::Elem>(i));
            }
        }
#endif
    }
};

int main() {
    using namespace at;
    namespace c = constants;

    Vec2D constexpr extents(c::bufLength, c::bufLength);

    PlatformHost platfHost;
    assert(a::getDevCount(platfHost) > 0);
    DevHost devHost = getDevByIdx(platfHost, 0);
    Platform platform;
    assert(a::getDevCount(platform) > 0);
    Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue(device);

    BufH2D<Elem> bufH = a::allocBuf<Elem, Idx>(devHost, extents);
    assert(a::getExtents(bufH).x() == a::getExtents(bufH)[1]);
    assert(a::getExtents(bufH).y() == a::getExtents(bufH)[0]);

    Vec2D const pitchesBytes = a::getPitchesInBytes(bufH);
    assert(pitchesBytes.x() == pitchesBytes[1]);
    assert(pitchesBytes.y() == pitchesBytes[0]);
    assert(pitchesBytes.x() == sizeof(Elem));
    assert(pitchesBytes.y() == extents.x() * sizeof(Elem));

    std::ranges::generate(std::span(bufH.data(), a::getExtents(bufH).prod()),
                          [i = 0]() mutable { return i--; });
    Elem const* const bufHData = bufH.data();
    assert(*bufHData == 0);
    assert(*(bufHData + 2) == -2);
    assert(*(bufHData + extents.x() * extents.y() - 1) == (-static_cast<int>(extents.prod()) + 1));
    assert(bufH[Vec2D(0, 0)] == *bufHData);
    assert(bufH[Vec2D(0, 11)] == *(bufHData + 11));
    assert(bufH[Vec2D(7, 0)] == *(bufHData + 7 * extents.x()));
    assert(bufH[Vec2D(extents.y() - 1, extents.x() - 1)] == *(bufHData + extents.prod() - 1));

    Buf2D<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufH));
    a::memcpy(queue, buf, bufH);
    Vec2D const pitches = a::getPitchesInBytes(buf);
    assert(pitches.x() == sizeof(Elem));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    assert(pitches.y() == extents.x() * sizeof(Elem));
#endif

    WorkDiv2D const workDiv = MakeWorkDiv<Acc2D>(extents);
    a::exec<Acc2D>(queue, workDiv, CheckKernel{}, buf.data(), a::getExtents(buf),
                   a::getPitchesInBytes(buf));
}
