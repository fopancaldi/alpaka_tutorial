Code I have used to display the most basic functionalities of [alpaka](https://github.com/alpaka-group/alpaka) while learning it.

Should contain everything a new user should know.

# Usage

    cmake -S . -B build -D ALPAKA_ACC_CONFIG=ON
    cmake --build build
    build/main

where `ALPAKA_ACC_CONFIG` is one of

- `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED`
- `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED` (requires Intel's [Thread Building Blocks](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html))
- `ALPAKA_ACC_GPU_CUDA_ENABLED` (requires a [cuda](https://developer.nvidia.com/cuda-toolkit)-capable machine)

# Dependencies

None.
