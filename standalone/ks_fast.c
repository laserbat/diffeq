// Finite-difference code for 2D Kuramoto-Sivashinsky equation.
// Uses O(h^2) finite-difference stencils and Euler's method.
//
// Compile with:
// $ gcc -Ofast -march=native -fopenmp ./ks.c -o ks -lm
// View the simulation using mpv:
// $ ./ks | mpv -
// Or use ffmpeg to convert to video format of choice.
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
// SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
// IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
// Olga U., me@lzr.pw

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define W (512)
#define H (512)

#define STENCIL_SIZE 3
#define BIH_STENCIL_SIZE 5

// Convenience defines for OpenMP, doesn't affect anything if OpenMP is
// disabled or unavoidable.
#define FOREACH_POINT \
    _Pragma("omp parallel for collapse(2) schedule(guided)")\
        for(int i = 0; i < H; i ++) for(int j = 0; j < W; j ++)

#define SECTIONS \
    _Pragma("omp sections")
#define SECTION \
    _Pragma("omp section")

// Helper macro for accessing the grid, taking correction and
// boundary wrapping into account
#define F(__dx, __dy) (grid[(x + H + __dx) % H][(y + W + __dy) % W])

// Discretization parameters
static const int SKIP_FRAMES = 10; // Steps between each output frame
static const double TIME_STEP = 0.01;
static const double GRID_STEP = 0.8;

// Coefficients for laplacian stencil
static const double L_COEFF[3] = {
    0, 1.0, -4.0
};

// Biharmonic
static const double B_COEFF[5] = {
    0, 1.0, 2.0, -8.0, 20.0
};

// Derivative
static const double D_COEFF[3] = {
    0, 1.0/2.0, -1.0/2.0
};

// Points on laplacian stencil
static const int L_STENCIL[STENCIL_SIZE][STENCIL_SIZE] = {
    {0, 1, 0},
    {1, 2, 1},
    {0, 1, 0},
};

// Points on biharmonic stencil
static const int B_STENCIL[BIH_STENCIL_SIZE][BIH_STENCIL_SIZE] = {
    {0, 0, 1, 0, 0},
    {0, 2, 3, 2, 0},
    {1, 3, 4, 3, 1},
    {0, 2, 3, 2, 0},
    {0, 0, 1, 0, 0},
};

// Points on derivative stencil
static const int D_STENCIL[STENCIL_SIZE][STENCIL_SIZE] = {
    {0, 1, 0},
    {0, 0, 0},
    {0, 2, 0},
};

// Coordinate shift to center the laplacian / derivative stencil
static const int SHIFT = (STENCIL_SIZE - 1) / 2;

// Same for biharmonic stencil
static const int BIH_SHIFT = (BIH_STENCIL_SIZE - 1) / 2;

// Values used by color() function to compute the shading, mostly arbitrary
static const double C_MUL = 2;
static const double C_SCALE[6][2] = {
    {0.05, 0.9},
    {0.003, 0.025},
    {0.02, 0.9},
    {0.001, 0.025},
    {0.02, 0.9},
    {0.007, 0.025}
};

// Fairly arbitrary function that maps grid values to a gradient
static void color(int x, int y, double world[H][W], char * c_val){
    double c = world[x][y];

    for(int n = 0; n < 3; n ++)
        c_val[n] = 0.5 + 255.0 * pow(
            sin(  1 + c * C_MUL * C_SCALE[2 * n + 0][0] * M_PI)
                * C_SCALE[2 * n][1] +
            cos(0.5 + c * C_MUL * C_SCALE[2 * n + 1][0] * M_PI)
                * C_SCALE[2 * n + 1][1], 2);
}

// Computes right-hand side of the equation
static inline double rhs(int x, int y, double grid[H][W]){
    double dx, dy, grad_sq;
    double laplacian, biharmonic;

    dx = dy = 0;
    laplacian = biharmonic = 0;
    for(int i = 0; i < BIH_STENCIL_SIZE; i ++){
        for(int j = 0; j < BIH_STENCIL_SIZE; j ++){
            double p = F(i - BIH_SHIFT, j - BIH_SHIFT);
            biharmonic += p * B_COEFF[B_STENCIL[i][j]];

            if (i >= STENCIL_SIZE || j >= STENCIL_SIZE) continue;

            p = F(i - SHIFT, j - SHIFT);
            dx += p * D_COEFF[D_STENCIL[i][j]];
            dy += p * D_COEFF[D_STENCIL[j][i]];

            laplacian += p * L_COEFF[L_STENCIL[i][j]];
        }
    }

    dx /= GRID_STEP;
    dy /= GRID_STEP;
    laplacian /= pow(GRID_STEP, 2);
    biharmonic /= pow(GRID_STEP, 4);

    grad_sq = pow(dx, 2) + pow(dy, 2);

    return -laplacian - biharmonic - 0.5 * grad_sq;
}

int main(void){
    static double grid[2][H][W];

    // Maximum possible length of PPM header
    const int MAX_HEADER_LEN = 128;

    // Entire PPM output
    char frame[W * H * 3 + MAX_HEADER_LEN];

    // Points to the actual bitmap part of output
    char *image;

    size_t header_len =
        snprintf(frame, MAX_HEADER_LEN,"P6\n%d %d\n255\n", W, H);

    size_t frame_len = 3 * W * H + header_len;

    // Image starts right after the header, so we just shift the pointer
    image = frame + header_len;

    // Initial conditions, an exponential bump in the center of grid
    FOREACH_POINT
        grid[0][i][j] = exp(-500 * hypot(i - H/2, j - W/2) / hypot(W, H));

    while(true) SECTIONS {
        // Output the grid
        SECTION {
            FOREACH_POINT
                color(i, j, grid[0], &image[3 * (i * W + j)]);

            fwrite(frame, 1, frame_len, stdout);
        }

        SECTION {
            for(int k = 0; k < SKIP_FRAMES; k ++) {
                FOREACH_POINT
                    grid[1][i][j] = rhs(i, j, grid[0]);

                FOREACH_POINT
                    grid[0][i][j] += TIME_STEP * grid[1][i][j];
            }
        }
    }

    return 0;
}
