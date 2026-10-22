#include "alpaka_tutorial.hpp"

#include <algorithm>
#include <alpaka/extent/Traits.hpp>
#include <cassert>
#include <utility>

int main() {
    namespace a = alpaka;
    namespace at = alpaka_tutorial;

    at::PlatformHost ph{};
    assert(a::getDevCount(ph) > 0);
    at::DevHost dh = a::getDevByIdx(ph, 0);
    at::QueueHost qh(dh);
    at::BufH1D<float> b1 = a::allocBuf<float, at::Idx>(dh, at::Vec1D{4}),
                      b2 = a::allocBuf<float, at::Idx>(dh, at::Idx{2});
    std::ranges::generate(b1, [f = 0.f]() mutable {
        const float r = f;
        f += 0.5f;
        return r;
    });
    std::ranges::generate(b2, [f = 0.f]() mutable {
        const float r = f;
        f -= 0.5f;
        return r;
    });

    using std::swap;
    swap(b1, b2);
    assert(a::getExtents(b1).x() == 2);
    assert(a::getExtents(b2).x() == 4);
    assert(std::ranges::all_of(b1, [f = 0.f](float f_) mutable {
        const bool r = f_ == f;
        f -= 0.5f;
        return r;
    }));
    assert(std::ranges::all_of(b2, [f = 0.f](float f_) mutable {
        const bool r = f_ == f;
        f += 0.5f;
        return r;
    }));
}
