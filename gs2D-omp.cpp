#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#include <stdio.h>
#include "utils.h"
using namespace std;
#include <math.h>

// Gauss-Seidel Method
void gaussSeidel(int N, double *u_gs){
    double ri = sqrt(N); // initial residual
    double r = ri + 1; // residual
    double sum = 0.0;
    int count = 0;
    double h = 1.0 / (N + 1.0);
    double diag = 2.0/pow(h, 2);
    double odiag = -1.0/pow(h, 2);

    int iter = 5000;

    if(N == 100000){
        iter = 100;
    }

    // count decides k+1, k
    while(count < iter && r > ri/(pow(10, 4))){
        // set residual to be 0.0
        r = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < N; i++){
            sum = 0.0;
            if(i == 0){
                sum += odiag * u_gs[i+1];
            }
            else if(i == N-1){
                sum += odiag * u_gs[i-1];
            }
            else{
                sum += odiag * u_gs[i-1] + odiag * u_gs[i+1];
            }
            u_gs[i] = 1.0 / diag * (1.0 - sum); 
        }

        #pragma omp parallel for reduction(+:r)
        for(int i = 0; i < N; i++){
            double au = 0;

            if(i == 0){
                au = diag * u_gs[0] + odiag * u_gs[1];
            }
            else if(i == N-1){
                au = odiag * u_gs[N-2] + diag * u_gs[N-1];
            }
            else{
                au = odiag * (u_gs[i-1] + u_gs[i+1]) + diag * u_gs[i];
            }
            r += pow((au - 1.0), 2);
        }
        r = sqrt(r);
        count++;
    }

    // output
    cout << "Gauss-Seidel Method: " << endl;
    // this is for printing the solution u
    // for(int i = 0; i < N; i++){
    //     cout << u_gs[i] << endl;
    // }
    cout << "initial residual: " << ri << endl;
    cout << "residual: " << r << endl;
    cout << "iterations: " << count << endl;
}


int main(int argc, char **argv)
{
    int N = 100000;
    Timer t;

    // declare and initialize u, u_gs
    double* u = (double *)malloc(N * sizeof(double));
    double* u_gs = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) u[i] = 0.0;
    for (int i = 0; i < N; i++) u_gs[i] = 0.0;

    // output format
    cout << " " << endl;
    cout << "—————— N = " << N << " ——————" << endl;
    cout << " " << endl;
    // output GS
    t.tic();
    gaussSeidel(N, u_gs);
    double timeG = t.toc();
    cout << "time = " << timeG << endl;
    cout << " " << endl;

    free(u);
    free(u_gs);
    return 0;
}