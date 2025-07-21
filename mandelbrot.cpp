#include "mandelbrot_kernel.cuh"
#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <numeric>
#include <complex>
#include <cmath>
#include <limits>

using DataType = double;

static int max_iterations = 5;
static DataType zoom = 1.0;
static DataType offsetX = 0.0;
static DataType offsetY = 0.0;

const int normal = 1;
const int WIDTH = 800;
const int HEIGHT = 600;

int isInSet(std::complex<DataType> c, int max_iterations)
{
    std::complex<DataType> z(0, 0);
    int i;
    for (int i = 0; i < max_iterations; i++)
    {
        z = std::pow(z, 2) + c;
        if (std::norm(z) > normal) 
        {
            return i;
        }   
    }
    return max_iterations;
}

template <typename T>
constexpr T lerp(T a, T b, T t) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(t)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return a + t * (b - a);
}

void drawFractal(SDL_Renderer* renderer)
{
    SDL_RenderClear(renderer);
    size_t mem_size = WIDTH * HEIGHT * sizeof(int);
    int* h_screen;
    cudaError(cudaMallocHost(&h_screen, mem_size));

    int* d_screen;
    cudaError(cudaMalloc(reinterpret_cast<void**>(&d_screen), mem_size));
    launch_calculateScreen(d_screen, zoom, offsetX, offsetY, max_iterations);
    cudaMemcpy(h_screen, d_screen, mem_size, cudaMemcpyDeviceToHost);

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            /*
            DataType point_x = lerp(-2.0f / zoom + offsetX, 2.0f / zoom + offsetX, static_cast<DataType>(x)/HEIGHT);
            DataType point_y = lerp(-2.0f / zoom + offsetY, 2.0f / zoom + offsetY, static_cast<DataType>(y)/WIDTH);
            */
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            //int iters = isInSet(std::complex<DataType>(point_x, point_y), max_iterations);
            if (h_screen[y * WIDTH + x] < max_iterations)
            {
                SDL_SetRenderDrawColor(renderer, 
                20 * h_screen[y * WIDTH + x] % 255,
                3 * h_screen[y * WIDTH + x] % 255, 
                10 * h_screen[y * WIDTH + x] % 255, 
                255);
            }
            SDL_RenderPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer("Mandelbrot",800, 600,0, &window, &renderer);
        if (!window || !renderer) {
        SDL_Log("Failed to create window/renderer: %s", SDL_GetError());
        return 1;
    }

    SDL_Event event;

    while (true) 
    {
        while (SDL_PollEvent(&event)) 
        {
            if (event.type == SDL_EVENT_QUIT) 
            {
                SDL_DestroyRenderer(renderer);
                SDL_DestroyWindow(window);
                SDL_Quit();
                return 0;
            }
            if (event.type == SDL_EVENT_KEY_DOWN) 
            {
                switch (event.key.scancode) 
                {
                    case SDL_SCANCODE_W:    offsetY -= 0.1 / zoom; break;
                    case SDL_SCANCODE_S:  offsetY += 0.1 / zoom; break;
                    case SDL_SCANCODE_A:  offsetX -= 0.1 / zoom; break;
                    case SDL_SCANCODE_D: offsetX += 0.1 / zoom; break;
                    case SDL_SCANCODE_Q:  zoom *= 1.1; break;
                    case SDL_SCANCODE_E: zoom /= 1.1; break;
                    case SDL_SCANCODE_R:  max_iterations += 1; break;
                    case SDL_SCANCODE_T: max_iterations -= 1; break;
                    default: break;
                }
            }
        }
        drawFractal(renderer);
    }
   
}