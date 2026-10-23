cmake_minimum_required(VERSION 3.25)
project(
  alpakaTutorial
  VERSION 0.0.1
  LANGUAGES CXX)

set(DESIRED_ALPAKA_VERSION
    2.1.1
    CACHE STRING "Desired alpaka version")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CheckLanguage)
include(FetchContent)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS
  " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual"
  " -Wformat=2 -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
  " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi"
  " -Woverloaded-virtual -Wnon-virtual-dtor -Wold-style-cast")
string(APPEND CMAKE_CXX_FLAGS_DEBUG
       " -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -g -pg")

set(alpaka_serial_flag "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
set(alpaka_tbb_flag "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
set(alpaka_cuda_flag "ALPAKA_ACC_GPU_CUDA_ENABLED")

foreach(flag IN ITEMS alpaka_serial_flag alpaka_tbb_flag alpaka_cuda_flag)
  if(${${flag}})
    if(NOT alpaka_acc_config)
      set(alpaka_acc_config ${${flag}})
      add_compile_definitions(${${flag}})
    else()
      message(FATAL_ERROR "Only one backend configuration should be enabled")
    endif()
  endif()
endforeach()

if(NOT alpaka_acc_config)
  message(FATAL_ERROR "Define one backend configuration")
elseif(alpaka_acc_config STREQUAL alpaka_tbb_flag)
  find_package(TBB REQUIRED)
  link_libraries(tbb)
elseif(alpaka_acc_config STREQUAL alpaka_cuda_flag)
  check_language(CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -O0 -g -pg -G")

  set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
  add_compile_options(--expt-relaxed-constexpr)
  add_compile_options(--extended-lambda)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(alpaka ${DESIRED_ALPAKA_VERSION} QUIET)
if(NOT alpaka_FOUND)
  FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
    GIT_TAG ${DESIRED_ALPAKA_VERSION})
  block()
  set(BUILD_TESTING OFF)
  FetchContent_MakeAvailable(alpaka)
  endblock()
endif()

add_library(alpakaTutorial INTERFACE)
target_include_directories(alpakaTutorial INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(alpakaTutorial INTERFACE alpaka::alpaka)
add_library(alpakaTutorial::alpakaTutorial ALIAS alpakaTutorial)
link_libraries(alpakaTutorial::alpakaTutorial)

add_subdirectory(tutorials)
