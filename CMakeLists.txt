cmake_minimum_required(VERSION 3.28)

# Every build covers both GPUs, sm_75 (the 2080 Ti) and sm_89 (the laptop), so
# each kernel keeps compiling for both. Pass -DCMAKE_CUDA_ARCHITECTURES to
# override it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 89)
endif()

project(Flint VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# The library and the executables share one folder, where the tests look for
# the library and Windows finds the DLL next to the executables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# Line info maps SASS back to source lines in Nsight Compute and
# compute-sanitizer reports, at no runtime cost.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Warnings shared by every Flint target. nvcc passes host-compiler flags on
# through -Xcompiler.
function(flint_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
  endif()
endfunction()

add_subdirectory(src)
add_subdirectory(bench)
