cmake_minimum_required(VERSION 3.25)
project(alpaka_tutorial VERSION 0.0.1 LANGUAGES CXX)

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
    " -O2 -march=native -ftree-vectorize -funroll-loops -funsafe-math-optimizations"
    " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual"
    " -Wformat=2 -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
    " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi"
    " -Woverloaded-virtual -Wnon-virtual-dtor -Wold-style-cast"
)
string(
    APPEND
    CMAKE_CXX_FLAGS_DEBUG
    " -fsanitize=address,undefined -fno-omit-frame-pointer -pg"
)
string(
    APPEND
    CMAKE_EXE_LINKER_FLAGS_DEBUG
    " -fsanitize=address,undefined -fno-omit-frame-pointer"
)

set(alpaka_serial_flag "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
set(alpaka_tbb_flag "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
set(alpaka_cuda_flag "ALPAKA_ACC_GPU_CUDA_ENABLED")

set(supported_backend_flags alpaka_serial_flag alpaka_tbb_flag alpaka_cuda_flag)
foreach(flag IN LISTS supported_backend_flags)
    if(${${flag}})
        if(NOT alpaka_acc_config)
            set(alpaka_acc_config ${${flag}})
            add_compile_definitions(${${flag}})
        else()
            message(
                FATAL_ERROR
                "Only one backend configuration should be enabled"
            )
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
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")

    set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES "50;60;61;62;70;80;90")
    add_compile_options("--expt-relaxed-constexpr")
endif()

# TODO: Understand why the address sanitizer does not behave well in the cnaf machine
if(alpaka_acc_config STREQUAL ${alpaka_cuda_flag})
    foreach(flags IN ITEMS CMAKE_CXX_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS_DEBUG)
        string(REPLACE " -fsanitize=address,undefined" "" ${flags} ${${flags}})
    endforeach()
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

file(GLOB all_cpp ${CMAKE_SOURCE_DIR}/cpp/*.cpp)
message("$all_cpp = ${all_cpp}")
foreach(cpp_file IN LISTS all_cpp)
    block()
        cmake_path(REMOVE_EXTENSION cpp_file OUTPUT_VARIABLE cpp_file_noext)
        cmake_path(GET cpp_file_noext FILENAME filename)
        add_executable(${filename} ${cpp_file})
        target_include_directories(${filename} PRIVATE ${CMAKE_SOURCE_DIR}/hpp)
    endblock()
    if(alpaka_acc_config STREQUAL ${alpaka_cuda_flag})
        set_source_files_properties(${cpp_file} PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()
