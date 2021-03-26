// Finite-difference code for 2D Kuramoto-Sivashinsky equation.
//
// Uses isotropic stencils for spatial discretization and
// RK4 for time stepping.
//
// Should have O(h^4) error in time and space.
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

#define W (128)
#define H (128)

// O(h^4) biharmonic stencil is 7x7, while laplacian and derivative are 5x5
#define STENCIL_SIZE 5
#define BIH_STENCIL_SIZE 7

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
#define F(__dx, __dy) (grid[(x + H + __dx) % H][(y + W + __dy) % W] +\
        mul * correction[(x + H + __dx) % H][(y + W + __dy) % W])

// Discretization parameters
static const int SKIP_FRAMES = 5; // Steps between each output frame
static const double TIME_STEP = 0.04;
static const double GRID_STEP = 0.8;

// Coefficients for Runge-Kutta (RK4) timestepping
static const double RK_COEFF[5] = {0, 0.5, 0.5, 1};

// Coefficients for O(h^4) laplacian and biharmonic taken from
//
// Patra, Michael & Karttunen, Mikko. (2006).
// Stencils with isotropic discretizationerror for differential operators.
// Numerical Methods for Partial Differential Equations.
// 22. 936 - 953. 10.1002/num.20129.

// Coefficients for laplacian stencil
static const double L_COEFF[6] = {
    0, -1.0/30.0, -1.0/60.0, 4.0/15.0, 13.0/15.0, -21.0/5.0
};

// Biharmonic
static const double B_COEFF[9] = {
    0, -17.0/180.0, 1.0/45.0, -29.0/180.0, 47.0/45.0,
    7.0/30.0, -187.0/90.0, -191.0/45.0, 779.0/45.0
};

// Coefficients for O(h^4) derivative computed using a method described in
//
// Anderberg, Tommy (2012):
// Gradient and smoothing stencils with isotropic discretization error.
// https://simplicial.net/hanlon/papers/stencils.pdf
//
// Paper above derrives a 3D stencil, but the same process works for 2D
//

static const double D_COEFF[6] = {
    0, 1.0/240.0, 3.0/40.0, -1.0/80.0, 1.0/24.0, -29.0/40.0
};

// Points on laplacian stencil
static const int L_STENCIL[STENCIL_SIZE][STENCIL_SIZE] = {
    {0, 1, 2, 1, 0},
    {1, 3, 4, 3, 1},
    {2, 4, 5, 4, 2},
    {1, 3, 4, 3, 1},
    {0, 1, 2, 1, 0},
};

// Points on biharmonic stencil
static const int B_STENCIL[BIH_STENCIL_SIZE][BIH_STENCIL_SIZE] = {
    {0, 0, 1, 2, 1, 0, 0},
    {0, 3, 4, 5, 4, 3, 0},
    {1, 4, 6, 7, 6, 4, 1},
    {2, 5, 7, 8, 7, 5, 2},
    {1, 4, 6, 7, 6, 4, 1},
    {0, 3, 4, 5, 4, 3, 0},
    {0, 0, 1, 2, 1, 0, 0},
};

// Points on derivative stencil
static const int D_STENCIL[STENCIL_SIZE][STENCIL_SIZE] = {
    { 0,  1,  2,  1,  0},
    { 3,  4,  5,  4,  3},
    { 0,  0,  0,  0,  0},
    {-3, -4, -5, -4, -3},
    { 0, -1, -2, -1,  0},
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
static inline double rhs(int x, int y, double grid[H][W], double mul,
        double correction[H][W]){
    double dx, dy, grad_sq;
    double laplacian, biharmonic;

    dx = dy = 0;
    laplacian = biharmonic = 0;
    for(int i = 0; i < BIH_STENCIL_SIZE; i ++){
        for(int j = 0; j < BIH_STENCIL_SIZE; j ++){
            if (i < STENCIL_SIZE && j < STENCIL_SIZE){
                dx += F(i - SHIFT, j - SHIFT)
                    * copysign(D_COEFF[abs(D_STENCIL[i][j])], D_STENCIL[i][j]);
                dy += F(j - SHIFT, i - SHIFT)
                    * copysign(D_COEFF[abs(D_STENCIL[i][j])], D_STENCIL[i][j]);

                laplacian  += F(i - SHIFT, j - SHIFT)
                    * L_COEFF[L_STENCIL[i][j]];
            }

            biharmonic += F(i - BIH_SHIFT, j - BIH_SHIFT)
                * B_COEFF[B_STENCIL[i][j]];
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
    static double grid[5][H][W];

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

        // Time-step with RK4
        SECTION {
            for(int k = 0; k < SKIP_FRAMES; k ++) {
                for(int p = 1; p < 5; p ++)
                    FOREACH_POINT
                        grid[p][i][j] = rhs(i, j, grid[0], RK_COEFF[p - 1]
                                * TIME_STEP, grid[p - 1]);

                FOREACH_POINT
                    grid[0][i][j] += (TIME_STEP/6.0) * (
                                grid[1][i][j] + 2 * grid[2][i][j] +
                            2 * grid[3][i][j] +     grid[4][i][j]
                        );
            }
        }
    }

    return 0;
}
