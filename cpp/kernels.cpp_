#include "alpaka_tutorial.hpp"

#include <algorithm>

namespace a = alpaka;
namespace at = alpaka_tutorial;

struct AccumulateKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, const at::Elem* buf, at::Elem* result,
                                  at::Idx bufLen) const {
        *result = 0;

        at::Elem& blockSum = a::declareSharedVar<at::Elem, __COUNTER__>(acc);
        blockSum = 0;
        a::syncBlockThreads(acc);

        for (at::Idx elemIdx : a::uniformElements(acc, bufLen)) {
            a::atomicAdd(acc, &blockSum, buf[elemIdx]);
        }
        if (a::oncePerBlock(acc)) {
            a::atomicAdd(acc, result, blockSum);
        }
    }
};

at::Buf0<at::Elem> Accumulate(at::Queue& queue, const at::Buf1<at::Elem>& buf) {
    using namespace at;
    const Vec1<Idx> ext = a::getExtents(buf);
    Buf0<Elem> result = a::allocAsyncBufIfSupported<Elem, Idx>(queue, Vec0<Idx>());
    a::WorkDivMembers<Dim1, Idx> workDiv = MakeWorkDiv<Acc1>(ext.x());
    a::exec<Acc1>(queue, workDiv, AccumulateKernel{}, buf.data(), result.data(), ext.x());
    return result;
}

int main() {
    using namespace at;
    namespace c = constants;

    PlatformH platfHost;
    DevH devHost = a::getDevByIdx(platfHost, 0);
    Platform platform;
    Device device = a::getDevByIdx(platform, 0);
    Queue queue(device);

    Buf1H<Elem> bufH = a::allocBuf<Elem, Idx>(devHost, c::bufLength);
    std::ranges::generate(std::span(bufH.data(), alpaka::getExtents(bufH).x()),
                          [i = 0]() mutable { return i++; });
    Buf1<Elem> buf = a::allocAsyncBufIfSupported<Elem, Idx>(queue, a::getExtents(bufH));
    a::memcpy(queue, buf, bufH);
    Buf0<Elem> result = Accumulate(queue, buf);
    Buf0H<Elem> resultH = a::allocBuf<Elem, Idx>(devHost, Vec0<Idx>());
    a::memcpy(queue, resultH, result);
    a::wait(queue);
    std::cout << "result: " << *result << '\n';
}
