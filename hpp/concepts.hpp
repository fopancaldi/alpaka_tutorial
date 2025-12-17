#pragma once

#include <alpaka/alpaka.hpp>
#include <type_traits>

namespace nostd {

template <typename T>
concept pointer = std::is_pointer_v<T>;

}

namespace noalpaka::concepts {

template <typename T>
concept Dim = std::integral<decltype(T::value)>;

template <typename T>
concept Dev = alpaka::isDevice<T>;

template <typename T>
concept Queue = alpaka::isQueue<T>;

} // namespace noalpaka::concepts

namespace alpaka_tutorial::concepts {

template <typename T>
concept Dim = noalpaka::concepts::Dim<T>;

template <typename T>
concept Dev = noalpaka::concepts::Dev<T>;

template <typename T>
concept Queue = noalpaka::concepts::Queue<T>;

template <typename T>
concept Acc = alpaka::concepts::Acc<T>;

template <typename T>
concept Acc1D = Acc<T> and (alpaka::Dim<T>::value == 1);

template <typename T>
concept Acc2D = Acc<T> and (alpaka::Dim<T>::value == 2);

template <typename T>
concept Buffer = requires(T t) {
    requires noalpaka::concepts::Dev<alpaka::Dev<T>>;
    typename alpaka::Elem<T>;
    requires noalpaka::concepts::Dim<alpaka::Dim<T>>;
    requires std::integral<alpaka::Idx<T>>;
    { alpaka::getPtrNative(t) } -> nostd::pointer;
};

} // namespace alpaka_tutorial::concepts
