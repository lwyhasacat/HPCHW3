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

// Jacobi Method
void Jacobi(int N, double *u){
    double* u_copy = (double *)malloc(N * sizeof(double)); // use u_copy to store u^k
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

        // store u_copy
        for(int i = 0; i < N; i++){
            u_copy[i] = u[i];
        }
        
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0; i < N; i++){
            double local_sum = 0.0;
            if(i == 0){
                local_sum += odiag * u_copy[i+1];
            }
            else if(i == N-1){
                local_sum += odiag * u_copy[i-1];
            }
            else{
                local_sum += odiag * u_copy[i-1] + odiag * u_copy[i+1];
            }
            // update u^(k+1)
            u[i] = 1.0 / diag * (1.0 - local_sum); 
            sum += local_sum;
        }

        // update residual
        #pragma omp parallel for reduction(+:r)
        for(int i = 0; i < N; i++){
            double au = 0;
            if(i == 0){
                au = diag * u[0] + odiag * u[1];
            }
            else if(i == N-1){
                au = odiag * u[N-2] + diag * u[N-1];
            }
            else{
                au = odiag * (u[i-1] + u[i+1]) + diag * u[i];
            }
            r += pow((au - 1.0), 2);
        }
        r = sqrt(r);

        // update u_copy
        for(int i = 0; i < N; i++){
            u_copy[i] = u[i];
        }
        count++;
    }

    // output
    cout << "Jacobi Method: " << endl;
    // this is for printing the solution u
    // for(int i = 0; i < N; i++){
    //     cout << u[i] << endl;
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
    // output Jacobi
    t.tic();
    Jacobi(N, u);
    double timeJ = t.toc();
    cout << "time = " << timeJ << endl;
    
    free(u);
    free(u_gs);
    return 0;
}