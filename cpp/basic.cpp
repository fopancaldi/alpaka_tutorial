#include "alpaka_tutorial.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace a = alpaka;
namespace at = alpaka_tutorial;

template <at::concepts::Queue TQueue, at::concepts::Buffer TBuf, typename TCheckFn>
    requires requires(TCheckFn f, int i) {
        { f(i) } -> std::convertible_to<typename a::Elem<TBuf>>;
    }
ALPAKA_FN_HOST void Check(TQueue& queue, TBuf buf, TCheckFn&& checkFn) {
    using Elem = std::remove_const_t<typename a::trait::ElemType<TBuf>::type>;

    at::PlatformHost platfHost;
    at::DevHost devHost = a::getDevByIdx(platfHost, 0);
    at::BufH1D<Elem> bufHost = a::allocBuf<Elem, at::Idx>(devHost, a::getExtents(buf));
    a::memcpy(queue, bufHost, buf);
    assert(std::ranges::all_of(bufHost, [&checkFn = std::as_const(checkFn), i = 0](Elem e) mutable {
        return e == checkFn(i++);
    }));
}

// NOTICE: in a kernel:
// - no dynamic memory allocation
// - no std. library containers
// - no exceptions
// - no recursion
// - only c++ features up to c++20
// - use alpaka::math, alpaka::atomic ..., alpaka::warp
struct Kernel {
    template <at::concepts::Acc1D TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, at::Elem const* in, at::Elem* out,
                                  at::Vec1D extents, at::Elem multiplier) const {
        // In this kernel it is not necessary to use groups, since those are needed only for
        // synchronization of elements in the same block
        for (at::Idx groupIdx : a::uniformGroups(acc, extents.x())) {
            for (a::ElementIndex<at::Idx> elemIdx :
                 a::uniformGroupElements(acc, groupIdx, extents.x())) {
                ALPAKA_ASSERT(elemIdx.global < extents.x());
                out[elemIdx.global] = in[elemIdx.global] * multiplier;
            }
        }
    }
};

int main() {
    using namespace at;
    namespace c = constants;

    // Platforms
    PlatformHost platfHost;
    assert(a::getDevCount(platfHost) > 0);
    DevHost devHost = getDevByIdx(platfHost, 0);
    Platform platform;
    std::vector<Device> devices = a::getDevs(platform);
    assert(a::getExtents(devices).x() > 0);
    Device device = devices.front();

    // Queues
    QueueHost queueHost(devHost);
    a::enqueue(queueHost, []() { std::this_thread::sleep_for(std::chrono::seconds(1)); });

    // Buffers + std::span
    BufH1D<Elem> bufH = a::allocBuf<Elem, Idx>(devHost, c::bufLength);
    std::ranges::generate(bufH, [i = 0]() mutable { return 2 * i++; });
    Check(queueHost, bufH, [](Elem e) { return 2 * e; });

    // Events + memcpy + asynchronous allocation
    Queue queue(device);
    Buf1D<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufH));
    a::memcpy(queue, buf, bufH);
    a::Event<Queue> endMemcpy(device);
    a::enqueue(queue, endMemcpy);
    a::wait(endMemcpy);
    Check(queue, buf, [](Elem e) { return 2 * e; });

    // Views + std::span
    ViewH1D<Elem> viewH(bufH.data(), a::getDev(bufH), a::getExtents(bufH));
    std::ranges::transform(std::span(viewH.data(), a::getExtents(viewH).x()), viewH.data(),
                           [](Elem e) { return e * e; });
    Check(queueHost, bufH, [](Elem e) { return 4 * e * e; });

    // Constant views
    a::ViewConst<BufH1D<Elem>> viewCH(bufH);
    Check(queueHost, viewCH, [](Elem e) { return 4 * e * e; });
    // The following line gives an error
    // viewCH[0] = -1;
    View1D<Elem const> viewC(buf.data(), a::getDev(buf), a::getExtents(buf));
    Check(queue, viewC, [](Elem e) { return 2 * e; });

    // Kernels
    Buf1D<Elem> buf2 = a::allocBuf<Elem, Idx>(device, a::getExtents(buf));
    WorkDiv1D const workDiv = MakeWorkDiv<Acc1D>(a::getExtents(buf));
    a::exec<Acc1D>(queue, workDiv, Kernel{}, buf.data(), buf2.data(), a::getExtents(buf), -1);
    a::wait(queue);
    Check(queue, buf2, [](Elem e) { return -2 * e; });
}
