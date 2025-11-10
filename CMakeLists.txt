cmake_minimum_required(VERSION 3.25)
project(alpaka_test VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CheckLanguage)
include(FetchContent)

string(
    APPEND
    CMAKE_CXX_FLAGS
    " -O2 -march=native -ftree-vectorize -funroll-loops -funsafe-math-optimizations"
    " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual"
    " -Wformat=2 -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
    " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi"
    " -Woverloaded-virtual -Wnon-virtual-dtor -Wold-style-cast"
)
string(
    APPEND
    CMAKE_CXX_FLAGS_DEBUG
    " -fsanitize=address,undefined -fno-omit-frame-pointer -g -pg"
)
string(
    APPEND
    CMAKE_EXE_LINKER_FLAGS_DEBUG
    " -fsanitize=address,undefined -fno-omit-frame-pointer"
)

set(SUPPORTED_BACKENDS
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
)
foreach(X IN LISTS SUPPORTED_BACKENDS)
    if(${X})
        if(NOT ALPAKA_ACC_CONFIG)
            set(ALPAKA_ACC_CONFIG ${X})
            add_compile_definitions(${X})
        else()
            message(
                FATAL_ERROR
                "Only one backend configuration should be enabled"
            )
        endif()
    endif()
endforeach()
if(NOT ALPAKA_ACC_CONFIG)
    message(FATAL_ERROR "Define one backend configuration")
endif()

if(ALPAKA_ACC_CONFIG STREQUAL "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
    link_libraries(tbb)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka
    GIT_TAG 2.0.0
)
set(alpaka_SOURCE_DIR "")
block(PROPAGATE alpaka_SOURCE_DIR)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(alpaka)
endblock()
include_directories(${alpaka_SOURCE_DIR}/include)

alpaka_add_executable(main main.cpp)
