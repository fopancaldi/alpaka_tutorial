#pragma once

#include <alpaka/alpaka.hpp>
#include <type_traits>

namespace nostd {

template <typename T>
concept pointer = std::is_pointer_v<T>;

}

namespace noalpaka::concepts {

template <typename T>
concept Dev = alpaka::isDevice<T>;

template <typename T>
concept Queue = alpaka::isQueue<T>;

template <typename T>
concept Acc = alpaka::isAccelerator<T>;

} // namespace noalpaka::concepts

namespace alpaka_tutorial::concepts {

template <typename T>
concept Acc1D = noalpaka::concepts::Acc<T> and (alpaka::Dim<T>::value == 1);

template <typename T>
concept Acc2D = noalpaka::concepts::Acc<T> and (alpaka::Dim<T>::value == 2);

template <typename T>
concept Buffer = requires(T t) {
    requires noalpaka::concepts::Dev<alpaka::Dev<T>>;
    typename alpaka::Elem<T>;
    requires std::integral<decltype(alpaka::Dim<T>::value)>;
    requires std::integral<alpaka::Idx<T>>;
    { alpaka::getPtrNative(t) } -> nostd::pointer;
};

} // namespace alpaka_tutorial::concepts
