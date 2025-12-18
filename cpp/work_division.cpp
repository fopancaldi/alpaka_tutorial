#include "alpaka_tutorial.hpp"

#include <cassert>
#include <stdexcept>

#define AT_CHECK_THROWS(cmd)                                                                       \
    do {                                                                                           \
        bool hasThrown = false;                                                                    \
        try {                                                                                      \
            cmd;                                                                                   \
        } catch (std::runtime_error const&) {                                                      \
            hasThrown = true;                                                                      \
        }                                                                                          \
        assert(hasThrown /**/&& #cmd && "did not throw");                                          \
    } while (false);

namespace a = alpaka;
namespace at = alpaka_tutorial;

struct Kernel {
    template <at::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&) const {}
};

template <at::concepts::Dim TDim>
void RunKernel(at::Queue& queue, at::Vec<TDim> blocks, at::Vec<TDim> blockThreads,
               at::Vec<TDim> threadElements) {
    at::WorkDiv<TDim> const workDiv(blocks, blockThreads, threadElements);
    a::exec<at::Acc<TDim>>(queue, workDiv, Kernel{});
    a::wait(queue);
}

int main() {
    using namespace at;

    Platform platform;
    assert(a::getDevCount(platform) > 0);
    Device device = a::getDevByIdx(platform, 0);
    Queue queue(device);

    // Multiple threads
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    RunKernel<Dim1D>(queue, 1, 1, 1);
    if constexpr (a::isSingleThreadAcc<Acc1D>) {
        AT_CHECK_THROWS(RunKernel<Dim1D>(queue, 1, 2, 1))
    } else {
        RunKernel<Dim1D>(queue, 1, 2, 1);
    }
#else
    AT_CHECK_THROWS(RunKernel<Dim1D>(queue, 1, 1025, 1))
#endif

    // Multiple elements
    RunKernel<Dim1D>(queue, 1, 1, 2);

    // Max grid size
    RunKernel<Dim2D>(queue, {1, 1 << 16}, {1, 1}, {1, 1});
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    RunKernel<Dim2D>(queue, {1 << 16, 1}, {1, 1}, {1, 1});
#else
    AT_CHECK_THROWS(RunKernel<Dim2D>(queue, {1 << 16, 1}, {1, 1}, {1, 1}))
#endif
}
