#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <numeric>
#include <complex>
#include <cmath>
#include <limits>


static int normal = 1;


static double zoom = 1.0;
static double offsetX = 0.0;
static double offsetY = 0.0;
static double multiplyer = 1.1;

int isInSet(std::complex<double> c, int max_iterations)
{
    std::complex<double> z(0, 0);
    int i;
    for (int i = 0; i < max_iterations; i++)
    {
        z = std::pow(z, 2) + c;
        if (std::norm(z) > normal) 
        {
            return i;
        }   
    }
    if (i == max_iterations) return max_iterations;
    return i + 1 - std::log(std::log2(std::norm(z)));
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
    int max_iterations = static_cast<int>(50 + 10 * std::log2(zoom)); 
    for (double x = 0.0; x < 1.0; x+=0.001)
    {
        //x *= multiplyer;
        for (double y = 0.0; y < 1.0; y+=0.001)
        {
            //y *= multiplyer;
            double point_x = lerp(-2.0 / zoom + offsetX, 2.0 / zoom + offsetX, x);
            double point_y = lerp(-2.0 / zoom + offsetY, 2.0 / zoom + offsetY, y);
            int iters = isInSet(std::complex<double>(point_x, point_y), max_iterations);
            if (iters == 0)
            {
                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                SDL_RenderDrawPointF(renderer, x * 800, y * 600);
            }
            else {
                SDL_SetRenderDrawColor(renderer, 
                30 * iters % 255,
                3 * iters % 255, 
                12 * iters % 255, 
                255);
                SDL_RenderDrawPointF(renderer, x * 800, y * 600);
            }
        }
    }

    SDL_RenderPresent(renderer);
        //SDL_Delay(2);
}

int main()
{
    SDL_Init(SDL_INIT_EVERYTHING);
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(800, 600,0, &window, &renderer);
    SDL_Event event;
    //SDL_RenderSetScale(renderer, 2, 2);


    while (true) 
    {
        while (SDL_PollEvent(&event)) 
        {
            if (event.type == SDL_QUIT) 
            {
                SDL_DestroyRenderer(renderer);
                SDL_DestroyWindow(window);
                SDL_Quit();
                return 0;
            }
            if (event.type == SDL_KEYDOWN) 
            {
                switch (event.key.keysym.sym) 
                {
                    case SDLK_w:    offsetY -= 0.1 / zoom; break;
                    case SDLK_s:  offsetY += 0.1 / zoom; break;
                    case SDLK_a:  offsetX -= 0.1 / zoom; break;
                    case SDLK_d: offsetX += 0.1 / zoom; break;
                    case SDLK_q:  zoom *= 1.1; break;
                    case SDLK_e: zoom /= 1.1; break;
                    case SDLK_r: {
                        if (normal > 0) 
                        {
                            normal -= 1; 
                        }
                        break;
                    }
                    case SDLK_t: normal += 1; break;

                }

                drawFractal(renderer);
            }
        }

        
    }
   
}