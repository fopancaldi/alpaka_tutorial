#include "types.hpp"

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

namespace alpakaTutorial::constants {

constexpr Idx size = 4;
constexpr Idx blockSize = 2;
constexpr Elem multiplier = -1;

} // namespace alpakaTutorial::constants

namespace a = alpaka;
namespace at = alpakaTutorial;

template <typename TBuf, typename TElem, typename Checker, typename TQueue>
    requires std::is_invocable_r_v<TElem, Checker, int>
ALPAKA_FN_HOST void Check(const TBuf& buf, Checker&& checker, TQueue& queue) {
    at::PlatformH platfH;
    at::DevH devH = a::getDevByIdx(platfH, 0);
    at::Buf1H<TElem> bufH = a::allocBuf<TElem, at::Idx>(devH, a::getExtents(buf));
    a::memcpy(queue, bufH, buf);
    assert(std::ranges::all_of(
        std::span(bufH.data(), a::getExtents(bufH).x()),
        [&checker = std::as_const(checker), i = 0](TElem e) mutable { return e == checker(i++); }));
}

// NOTICE: in a kernel:
// - no dynamic memory allocation
// - no std. library containers
// - no exceptions
// - no recursion
// - only c++ features up to c++20
// - use alpaka::math, alpaka::atomic ..., alpaka::warp
struct Kernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, const at::Elem* __restrict__ in, at::Elem* out,
                                  at::Idx size, at::Elem multiplier) const {
        // In this kernel it is not necessary to use groups, since the number of blocks is chosen at
        // runtime in order to guarantee that all problem space is covered
        for (at::Idx groupIdx : a::uniformGroups(acc, size)) {
            for (a::ElementIndex<at::Idx> elemIdx : a::uniformGroupElements(acc, groupIdx, size)) {
                if (elemIdx.global < size) {
                    out[elemIdx.global] = in[elemIdx.global] * multiplier;
                }
            }
        }
    }
};

int main() {
    using namespace at;
    namespace c = constants;

    // Platforms
    PlatformH platfHost;
    DevH devHost = getDevByIdx(platfHost, 0);
    Platform platform;
    std::vector<Device> devices = a::getDevs(platform);
    assert(a::getExtents(devices).x() == 1);
    Device device = devices.front();

    // Queues
    QueueH queueH(device);
    a::enqueue(queueH, []() { std::cout << "Queued work\n"; });

    // Buffers + std::span
    Buf1H<Elem> bufH = alpaka::allocBuf<Elem, Idx>(device, c::size);
    std::ranges::generate(std::span(std::data(bufH), getExtents(bufH).x()),
                          [i = 0]() mutable { return 2 * i++; });
    Check<Buf1<Elem>, Elem>(bufH, [](Elem e) { return 2 * e; }, queueH);

    // Events + memcpy + asynchronous allocation
    Queue queue(device);
    Buf1<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufH));
    a::memcpy(queue, buf, bufH);
    a::Event<Queue> endMemcpy(device);
    a::enqueue(queue, endMemcpy);
    a::wait(endMemcpy);
    Check<Buf1<Elem>, Elem>(bufH, [](Elem e) { return 2 * e; }, queue);

    // Views + std::span
    a::ViewPlainPtr<DevH, Elem, Dim1, Idx> viewH(std::data(bufH), devHost, a::getExtents(bufH));
    std::ranges::transform(std::span(std::data(viewH), a::getExtents(viewH).x()), std::data(viewH),
                           [](Elem e) { return e * e; });
    Check<Buf1<Elem>, Elem>(bufH, [](Elem e) { return 4 * e * e; }, queue);

    // Constant views
    a::ViewConst<Buf1H<Elem>> viewCH(bufH);
    Check<a::ViewConst<Buf1H<Elem>>, Elem>(viewCH, [](Elem e) { return 4 * e * e; }, queue);
    // The following line results in a compilation error
    // cViewAcc[0] = -1;

    // Kernels
    Buf1<Elem> buf2 = a::allocBuf<Elem, Idx>(device, a::getExtents(bufH));
    a::WorkDivMembers<Dim1, Idx> grid(a::core::divCeil(c::size, c::blockSize), Idx{1},
                                      c::blockSize);
    a::exec<Acc1>(queue, grid, Kernel{}, buf.data(), buf2.data(), a::getExtents(buf).x(),
                  c::multiplier);
    a::wait(queue);
    Check<Buf1<Elem>, Elem>(buf2, [](Elem e) { return -2 * e; }, queue);
}
