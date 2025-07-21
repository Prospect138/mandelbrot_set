
#include "mandelbrot_kernel.cuh"

#include <cuda_runtime.h>
#include <complex>
#include <cuda/std/complex>

using DataType = double;

const int normal = 1;

const int WIDTH = 800;
const int HEIGHT = 600;

__device__ double isInSet(cuda::std::complex<DataType> c, int max_iterations)
{
    cuda::std::complex<DataType> z(0.0f, 0.0f);
    for (int i = 0; i < max_iterations; i++)
    {
        z = cuda::std::pow(z, 2) + c;
        if (cuda::std::norm(z) > 4.0f) {
            float mu = i - log2f(log2f(norm(z)) * 0.5f);  // или просто:
            // float mu = i - log2f(log2f(norm(z)) / 2.0f);
            return mu;
        }
    }
    return max_iterations;
}

template <typename T>
__device__ constexpr T lerp(T a, T b, T t) {
    if (isnan(a) || isnan(b) || isnan(t)) return -99;
    return a + t * (b - a);
}

__global__ void calculateScreen(DataType* d_screen, DataType zoom, DataType offsetX, DataType offsetY, int max_iterations)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    DataType point_x = lerp(-2.0f / zoom + offsetX, 2.0f / zoom + offsetX, static_cast<DataType>(x)/WIDTH);
    DataType point_y = lerp(-1.5f / zoom + offsetY, 1.5f / zoom + offsetY, static_cast<DataType>(y)/HEIGHT);
    DataType iters = isInSet(cuda::std::complex<DataType>(point_x, point_y), max_iterations);
    d_screen[y * WIDTH + x] = iters;
}

void launch_calculateScreen(DataType* d_screen, float zoom, float offsetX, float offsetY, int max_iterations)
{
    dim3 threadBlock = {16, 16};
    dim3 grid = {(WIDTH + threadBlock.x - 1) / threadBlock.x, (HEIGHT + threadBlock.y - 1) / threadBlock.y};
    calculateScreen<<<grid, threadBlock>>>(d_screen, zoom, offsetX, offsetY, max_iterations);
    cudaDeviceSynchronize();
}

/*       

        grid.x (WIDTH = 800)
<-----------------------> 
threadBlock.x(16)
<--->
________________________
|___|___|___|___|___|___|  /\  1 threadBlock.y(16)
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  | grid.y (608 = (38 * 16 > HEIGHT (600) ? grid.y))
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  |
|___|___|___|___|___|___|  v
                            
*/                             