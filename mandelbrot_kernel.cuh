#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

void launch_calculateScreen(int* d_screen, float zoom, float offsetX, float offsetY, int max_iterations);

#endif // KERNEL_CUH