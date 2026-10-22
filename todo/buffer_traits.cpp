#include "alpaka_tutorial.hpp"

#include <concepts>
#include <ranges>
#include <type_traits>

template <typename T>
concept alpaka_buffer = requires(std::remove_cvref_t<T> t) {
    requires noalpaka::concepts::Dev<alpaka::Dev<std::remove_cvref_t<T>>>;
    requires noalpaka::concepts::Dim<alpaka::Dim<std::remove_cvref_t<T>>>;
    requires std::integral<alpaka::Idx<std::remove_cvref_t<T>>>;
    typename alpaka::Elem<std::remove_cvref_t<T>>;
    { alpaka::getDev(t) } -> noalpaka::concepts::Dev;
    { alpaka::getExtents(t) } /* -> TODO: constraint */;
    { alpaka::getPtrNative(t) } -> nostd::pointer;
    { alpaka::getOffsets(t) } /* -> TODO: constraint */;
};

template <typename T>
concept alpaka_1d_buffer = alpaka_buffer<T> and (alpaka::Dim<T>::value == 1);

template <typename T>
concept stl_container = requires(T t) {
    requires std::ranges::range<T>;
    typename T::value_type;
    typename T::size_type;
    /* typename T::allocator_type; */
    typename T::iterator;
    typename T::const_iterator;
    { t.size() } -> std::unsigned_integral;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.cbegin() } -> std::same_as<typename T::const_iterator>;
    { t.cend() } -> std::same_as<typename T::const_iterator>;
};

int main() {
    /* static_assert(alpaka_buffer<const alpaka_tutorial::BufH1D<float>>);
    static_assert(stl_container<std::vector<float>>);
    static_assert(stl_container<std::array<float, 5>>);

    static_assert(std::ranges::contiguous_range<alpaka_tutorial::BufH1D<float>>);
    static_assert(
        std::is_same_v<std::ranges::range_value_t<alpaka_tutorial::BufH1D<float>>, float>);

    alpaka_tutorial::BufH1D<float> b =
        alpaka::allocBuf<float, uint32_t>(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), 1);
    auto it = b.cbegin();

    alpaka_tutorial::BufH2D<float> b2 = alpaka::allocBuf<float, uint32_t>(
        alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0), alpaka_tutorial::Vec2D{2, 2});
    static_assert(
        std::is_same_v<std::ranges::range_value_t<alpaka_tutorial::BufH2D<float>>, float>); */
}
