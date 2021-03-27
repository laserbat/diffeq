#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <arkode/arkode_arkstep.h>
//#include <arkode/arkode_bandpre.h>
#include <nvector/nvector_openmp.h>
#include <sunlinsol/sunlinsol_band.h>
//#include <sunlinsol/sunlinsol_pcg.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sundials/sundials_types.h>

#define STENCIL_SIZE 5
#define BIH_STENCIL_SIZE 7

static const double TIME_STEP = 0.3;
static const double GRID_STEP = 0.8;
static const double GS2 = 1.0 / (GRID_STEP * GRID_STEP);

static const sunindextype S = 1024;
static const sunindextype N = S * S;
static const int THREADS = 8;

static const realtype REL_TOL = 1.e-8;
static const realtype ABS_TOL = 1.e-8;

static const double BL_COEFF[9] = {
    0, (-17.0/180.0) * GS2 * GS2, (1.0/45.0) * GS2 * GS2, (-29.0/180.0) * GS2 * GS2,
    ((47.0/45.0) * GS2 - 1.0/30.0) * GS2,
    ((7.0/30.0) * GS2 - 1.0/60.0) * GS2,
    ((-187.0/90.0) * GS2 + 4.0/15.0) * GS2,
    ((-191.0/45.0) * GS2 + 13.0/15.0) * GS2,
    ((779.0/45.0) * GS2 -21.0/5.0) * GS2
};

// Derivative
static const double D_COEFF[11] = {
    29.0/40.0, -1.0/24.0, 1.0/80.0, -3.0/40.0, -1.0/240.0,
    0,
    1.0/240.0, 3.0/40.0, -1.0/80.0, 1.0/24.0, -29.0/40.0
};

static const int BL_STENCIL[BIH_STENCIL_SIZE][BIH_STENCIL_SIZE] = {
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

static const double C_MUL = 5;
static const double C_SCALE[6][2] = {
    {0.05, 0.9},
    {0.003, 0.025},
    {0.02, 0.9},
    {0.001, 0.025},
    {0.02, 0.9},
    {0.007, 0.025}
};

static int rhs_ex(realtype t, N_Vector y_vec, N_Vector ydot_vec, void *user_data){
    realtype *y, *ydot;
    double dx, dy;

    y = N_VGetArrayPointer(y_vec);
    ydot = N_VGetArrayPointer(ydot_vec);

    for(int ii = 0; ii < S; ii ++){
        for(int jj = 0; jj < S; jj ++){
            dx = dy = 0;

            for(int i = 0; i < STENCIL_SIZE; i ++){
                for(int j = 0; j < STENCIL_SIZE; j ++){
                    int coord = (ii + i + S - SHIFT) % S;
                    coord *= S;
                    coord += (jj + j + S - SHIFT) % S;

                    dx += y[coord] * D_COEFF[5 + D_STENCIL[i][j]];
                    dy += y[coord] * D_COEFF[5 + D_STENCIL[j][i]];
                }
            }

            dx /= GRID_STEP;
            dy /= GRID_STEP;

            ydot[ii * S + jj] = -0.5 * (dx * dx + dy * dy);
        }
    }

    return 0;
}

static int rhs_im(realtype t, N_Vector y_vec, N_Vector ydot_vec, void *user_data){
    realtype *y, *ydot;
    double out;

    y = N_VGetArrayPointer(y_vec);
    ydot = N_VGetArrayPointer(ydot_vec);

    for(int ii = 0; ii < S; ii ++){
        for(int jj = 0; jj < S; jj ++){
            out = 0;

            for(int i = 0; i < BIH_STENCIL_SIZE; i ++){
                for(int j = 0; j < BIH_STENCIL_SIZE; j ++){
                    int coord;

                    coord = (ii + i + S - BIH_SHIFT) % S;
                    coord *= S;
                    coord += (jj + j + S - BIH_SHIFT) % S;
                    out += y[coord] * BL_COEFF[BL_STENCIL[i][j]];
                }
            }

            ydot[ii * S + jj] = -out;
        }
    }

    return 0;
}

int main(void){
    N_Vector gridvec = N_VNew_OpenMP(N, THREADS);
    realtype *grid_data = N_VGetArrayPointer_OpenMP(gridvec);
    SUNMatrix matrix = SUNBandMatrix(N, 1, 1);

    fprintf(stderr, "Data initialized\n");

    srand48(1753111);

    for(int i = 0; i < N; i ++){
        int x  = i % S;
        int y = i / S;
        grid_data[i] = exp(-100 * hypot(x - S/2.0, y - S/2.0)/hypot(S, S));;
    }

    void *ark = ARKStepCreate(rhs_ex, rhs_im, 0, gridvec);
    SUNLinearSolver lin_solver = SUNLinSol_Band(gridvec, matrix);
    //SUNLinearSolver lin_solver = SUNLinSol_PCG(gridvec, PREC_LEFT, 15);

    fprintf(stderr, "Linear solver initialized\n");

    ARKStepSStolerances(ark, REL_TOL, ABS_TOL);
    ARKStepSetLinearSolver(ark, lin_solver, matrix);
    ARKStepSetLinear(ark, 0);
    //ARKBandPrecInit(ark, N, 1, 1);
    //ARKStepSetDiagnostics(ark, stderr);
    ARKStepSetOrder(ark, 5);
    ARKStepSetOptimalParams(ark);

    fprintf(stderr, "Set up solver\n");

    realtype t, tret;
    t = TIME_STEP;

    int r = 0;
    while (r == 0) {
        r = ARKStepEvolve(ark, t, gridvec, &tret, ARK_NORMAL);
        fprintf(stderr, "Time step @%lf\n", t);
        t += TIME_STEP;
        printf("P6\n%d %d\n255\n", (int)S, (int)S);

        for(int i = 0; i < N; i ++){
            double cell = grid_data[i];
            for(int n = 0; n < 3; n ++){
                char color = 0.5 + 255.0 * pow(
                    sin(  1 + cell * C_MUL * C_SCALE[2 * n + 0][0] * M_PI)
                        * C_SCALE[2 * n][1] +
                    cos(0.5 + cell * C_MUL * C_SCALE[2 * n + 1][0] * M_PI)
                        * C_SCALE[2 * n + 1][1], 2);
                putchar(color);
            }
        }
    }

    return 0;
}
