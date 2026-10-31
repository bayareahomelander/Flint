# The Flint runtime library. It links only the static CUDA runtime: cuBLAS,
# NVML, and PyTorch belong to the tests and benchmarks, never to this target.
add_library(flint SHARED
  api/copy.cpp
  api/gemm.cpp
  api/status.cpp
  common/error.cpp
  kernels/copy.cu
  kernels/gemm.cu
)
target_include_directories(flint
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(flint PRIVATE
  FLINT_BUILD
  FLINT_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(flint PRIVATE CUDA::cudart_static)

# Only functions marked FLINT_API are exported, on Linux as on Windows, so a
# missing mark fails the tests on both machines.
set_target_properties(flint PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(NOT WIN32)
  # Keeps the static CUDA runtime's symbols out of the export table, so they
  # can't interpose with the copy PyTorch loads into the same process.
  target_link_options(flint PRIVATE LINKER:--exclude-libs,ALL)
endif()

flint_set_warnings(flint)
