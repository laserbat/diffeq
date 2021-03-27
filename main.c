#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cvode/cvode.h>
#include <cvode/cvode_diag.h>
#include <cvode/cvode_bandpre.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

#define STENCIL_SIZE 5
#define BIH_STENCIL_SIZE 7

static const double TIME_STEP = 0.3;
static const double GRID_STEP = 0.8;
static const double GS2 = 1.0 / (GRID_STEP * GRID_STEP);

static const sunindextype S = 100;
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
static const int D_STENCIL[BIH_STENCIL_SIZE][BIH_STENCIL_SIZE] = {
    { 0,  0,  0,  0,  0,  0, 0},
    { 0,  0,  1,  2,  1,  0, 0},
    { 0,  3,  4,  5,  4,  3, 0},
    { 0,  0,  0,  0,  0,  0, 0},
    { 0, -3, -4, -5, -4, -3, 0},
    { 0,  0, -1, -2, -1,  0, 0},
    { 0,  0,  0,  0,  0,  0, 0},
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

static int rhs(realtype t, N_Vector y_vec, N_Vector ydot_vec, void *user_data){
    realtype *y, *ydot;
    double out, dx, dy;

    y = N_VGetArrayPointer(y_vec);
    ydot = N_VGetArrayPointer(ydot_vec);

    for(int ii = 0; ii < S; ii ++){
        for(int jj = 0; jj < S; jj ++){
            out = 0;
            dx = dy = 0;

            for(int i = 0; i < BIH_STENCIL_SIZE; i ++){
                for(int j = 0; j < BIH_STENCIL_SIZE; j ++){
                    int coord;
                    coord = (ii + i + S - BIH_SHIFT) % S;
                    coord *= S;
                    coord += (jj + j + S - BIH_SHIFT) % S;
                    out += y[coord] * BL_COEFF[BL_STENCIL[i][j]];
                    dx += y[coord] * D_COEFF[5 + D_STENCIL[i][j]];
                    dy += y[coord] * D_COEFF[5 + D_STENCIL[j][i]];
                }

            }

            dx /= GRID_STEP;
            dy /= GRID_STEP;
            out += 0.5 * (dx * dx + dy * dy);

            ydot[ii * S + jj] = -out;
        }
    }

    return 0;
}

int main(void){
    N_Vector gridvec = N_VNew_Serial(N);
    realtype *grid_data = N_VGetArrayPointer_Serial(gridvec);

    fprintf(stderr, "Data initialized\n");

    srand48(1753111);

    for(int i = 0; i < N; i ++){
        int x  = i % S;
        int y = i / S;
        grid_data[i] = exp(-500 * hypot(x - S/2.0, y - S/2.0)/hypot(S, S));;
    }

    for(int i = 0; i < N; i ++){
        int x  = i % S;
        int y = i / S;
        grid_data[i] += grid_data[(i + 1) % N];
        grid_data[i] += grid_data[(i + N - 1) % N];
        grid_data[i] += grid_data[(i + S) % N];
        grid_data[i] += grid_data[(i + N - S) % N];
        grid_data[i] /= 5;
    }


    void *cv = CVodeCreate(CV_BDF);
    CVodeInit(cv, rhs, 0, gridvec);

    CVodeSStolerances(cv, REL_TOL, ABS_TOL);

    fprintf(stderr, "Set up solver\n");

    //SUNLinearSolver linsol = SUNLinSol_SPGMR(gridvec, PREC_LEFT, 15);
    //CVodeSetLinearSolver(cv, linsol, NULL);

    SUNNonlinearSolver nonlinsol = SUNNonlinSol_Newton(gridvec);
    CVodeSetNonlinearSolver(cv, nonlinsol);
    CVDiag(cv);

    //CVBandPrecInit(cv, N, 1, 1);

    realtype t, tret;
    t = TIME_STEP;

    int r = 0;
    while (r == 0) {
        r = CVode(cv, t, gridvec, &tret, CV_NORMAL);
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
