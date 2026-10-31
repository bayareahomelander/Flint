# flint_bench, the timing harness. Only the benchmarks link NVML, the driver
# API (to find the GPU before creating a context), and cuBLAS and cuBLASLt (the
# gemm baselines); the library never does.
add_executable(flint_bench
  cli.cpp
  copy_bench.cpp
  gemm_bench.cpp
  harness.cpp
  main.cpp
  mma_bench.cpp
  mma_kernels.cu
  monitor.cpp
  report.cpp
)
target_link_libraries(flint_bench PRIVATE flint CUDA::cudart_static CUDA::cuda_driver CUDA::nvml CUDA::cublas CUDA::cublasLt)
target_compile_definitions(flint_bench PRIVATE
  FLINT_BENCH_OUT_DIR="${PROJECT_BINARY_DIR}/bench-results"
)

# git_info.h records the commit the benchmark was built from, and whether the
# tree had uncommitted changes. The script runs on every build but rewrites the
# header only when it changes.
set(git_info_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_custom_target(flint_git_info
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DOUTPUT=${git_info_dir}/git_info.h -P ${PROJECT_SOURCE_DIR}/cmake/git_info.cmake
  BYPRODUCTS ${git_info_dir}/git_info.h
  VERBATIM
)
add_dependencies(flint_bench flint_git_info)
target_include_directories(flint_bench PRIVATE ${git_info_dir})

flint_set_warnings(flint_bench)
