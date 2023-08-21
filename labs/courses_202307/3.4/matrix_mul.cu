#include <stdio.h>
#include <math.h>
#include <iostream>
#include "error.cuh"

#define M 1000
#define N 500
#define K 1000 
#define BLOCK_SIZE 16

__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c_cpu[M*K];
__managed__ int c_gpu[M*K];

void cpu_matrix(int *a, int *b, int *c, int m, int n, int k){
    for(int i=0; i<m; ++i){
        for(int j=0; j<k; ++j){
            int sum_val = 0;
            for(int step=0; step<n; ++step){
                sum_val += a[i*n + step] * b[step*k + j];
            }
            c[i*k + j]=sum_val;
        }
    }
}

__global__ void gpu_matrix(int *a, int *b, int *c, int m, int n, int k){
    

    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int temp = 0;
    int idx;
    for(int step = 0; step < n/BLOCK_SIZE; ++step){

        int step_x = step*BLOCK_SIZE + threadIdx.x;
        int step_y = y;
        idx = step_y * n + step_x;
        
        if(step_y<m && step_x<n){
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }else{
            sub_a[threadIdx.y][threadIdx.x] = 0; 
        }

        step_x = x;
        step_y = step * BLOCK_SIZE + threadIdx.y;
        idx = step_y*k + step_x;
        if(step_x<k && step_y<n){
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }else{
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for(int i=0; i<BLOCK_SIZE; ++i){
            temp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(x<k && y<m){
        c[y*k + x] = temp;
    }
}

int main(){
    cudaEvent_t start, stop_cpu, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventCreate(&stop_gpu));

    for(int i = 0; i<M; ++i){
        for(int j=0; j<N; ++j){
            a[i*N + j] = rand() % 1024;
        }
    }

    for(int i=0; i<N; ++i){
        for(int j=0; j<K; ++j){
            b[i*K + j] = rand() % 1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE - 1) /BLOCK_SIZE;
    unsigned int grid_y = (K + BLOCK_SIZE - 1) /BLOCK_SIZE;

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    CHECK(cudaEventRecord(start));

    gpu_matrix<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    CHECK(cudaEventRecord(stop_gpu));

    CHECK(cudaEventSynchronize(stop_gpu));
    cpu_matrix(a, b, c_cpu, M, N, K);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    
    float time_cpu, time_gpu;
    bool errors = false;
    
    for(int i =0; i<M; ++i){
        for(int j=0; j<K; ++j){
            printf("cpu_val: %d, gpu_val: %d\n", c_cpu[i*K+j], c_gpu[i*K+j]);
            if(fabs(c_cpu[i*K+j] - c_gpu[i*K+j])>(1.0e-10)){
                errors = true;
                printf("cpu_val: %d, gpu_val: %d", c_cpu[i*K+j], c_gpu[i*K+j]);
            }
        }
    }

    CHECK(cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu));
    CHECK(cudaEventElapsedTime(&time_gpu, start, stop_gpu));

    printf(errors? "fail" : "pass");
    printf("cpu_time: %.2f   gpu_time: %.2f", time_cpu, time_gpu);
    return 0;
}


