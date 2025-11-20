#include "alpaka_tutorial.hpp"

namespace a = alpaka;
namespace at = alpaka_tutorial;

struct AccumulateKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&, const at::Elem* ptr, at::Elem* result) const {}
};

at::Buf0<at::Elem> Accumulate(at::Queue& queue, const at::Buf1<at::Elem>& buf) {
    using namespace at;
    DevH devHost = a::getDevByIdx(PlatformH(), 0);
    Buf0<Elem> result = a::allocAsyncBufIfSupported<Elem, Idx>(queue, Vec0<Idx>());

    return result;
}

int main() {}
