#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <arkode/arkode_arkstep.h>
#include <nvector/nvector_openmp.h>
#include <sunlinsol/sunlinsol_lapackband.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sundials/sundials_types.h>

static const sunindextype LBAND = 1;
static const sunindextype UBAND = 1;
static const sunindextype BANDWIDTH = 1 + LBAND + UBAND;

static const sunindextype S = 3;
static const sunindextype N = S * S;
static const int THREADS = 8;

static const realtype rel_tol = 1.e-6;
static const realtype abs_tol = 1.e-10;

static int rhs_ex(realtype t, N_Vector y_vec, N_Vector ydot_vec, void *user_data){
    realtype *y, *ydot;

    y = N_VGetArrayPointer(y_vec);
    ydot = N_VGetArrayPointer(ydot_vec);
    return 0;
}

static int rhs_im(realtype t, N_Vector y_vec, N_Vector ydot_vec, void *user_data){
    realtype *y, *ydot;

    y = N_VGetArrayPointer(y_vec);
    ydot = N_VGetArrayPointer(ydot_vec);
    return 0;
}

int main(void){
    N_Vector gridvec = N_VNew_OpenMP(N, THREADS);
    realtype *grid_data = N_VGetArrayPointer_OpenMP(gridvec);
    SUNMatrix matrix = SUNBandMatrix(N, LBAND, UBAND);
    realtype **matrix_data = SUNBandMatrix_Cols(matrix);

    for(int i = 0; i < N; i ++){
        grid_data[i] = 0;
    }

    for(int i = 0; i < N; i ++){
        for(int j = 1; j <= BANDWIDTH; j ++){
            matrix_data[i][j] = j;
        }
    }

    SUNBandMatrix_Print(matrix, stdout);


    void *ark = ARKStepCreate(rhs_ex, rhs_im, 0, gridvec);
    SUNLinearSolver lin_solver = SUNLinSol_LapackBand(gridvec, matrix);
    ARKStepSetLinearSolver(ark, lin_solver, matrix);
    ARKStepSetLinear(ark, 0);

    return 0;
}
