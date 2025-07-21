cmake_minimum_required(VERSION 3.20)

project(mandelbrot LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED sdl3)

set(CMAKE_CUDA_ARCHITECTURES 86 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(mandelbrot mandelbrot.cpp mandelbrot.cu)

target_compile_options(mandelbrot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(mandelbrot PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(mandelbrot PRIVATE
    ${SDL3_LIBRARIES}
    CUDA::cuda_driver
)

target_include_directories(mandelbrot PRIVATE ${SDL3_INCLUDE_DIRS})

set_target_properties(mandelbrot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)