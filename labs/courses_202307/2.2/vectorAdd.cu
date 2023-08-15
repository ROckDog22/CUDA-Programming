#include <math.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void add(const double *A, const double *B, double *C, int N)
{       
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx<N){
                C[idx] = A[idx] + B[idx];
        }
}

void add_cpu(const double *A, const double *B, double *C, int N)
{
        for(int i=0; i<N; ++i)
        {
                C[i]=A[i]+B[i];
        }
}

void checkvalue(const double *z, int N)
{
        bool flag = true;
        for(int i =0 ;i<N; ++i)
        {
                if(fabs(z[i] - 3)>1.0e-5)
                {       
                        flag = false;
                }
        }
        printf("%s\n", flag? "pass":"fail");
}
int main(){
        int N = 100000000;
        int M = sizeof(double) * N;
        double* h_x, *h_y, *h_z;
        h_x = (double*)malloc(M);
        h_y = (double*)malloc(M);        
        h_z = (double*)malloc(M);
        for(int i=0; i<N; ++i)
        {
                h_x[i] = 1.0;
                h_y[i] = 2.0;
        }

        double *d_x, *d_y, *d_z;
        cudaMalloc((void **)&d_x, M);
        cudaMalloc((void **)&d_y, M);
        cudaMalloc((void **)&d_z, M);
        cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

        dim3 block_size(256);
        dim3 grid_size((N+block_size.x-1)/ block_size.x);
        // add<<<grid_m, block_m>>>(h_a, h_b, h_c, N);

        // const int block_size = 128;
        // const int grid_size = (N + block_size - 1) / block_size;
        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
        checkvalue(h_z, N);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        free(h_x);
        free(h_y);
        free(h_z);
        return 0;
}