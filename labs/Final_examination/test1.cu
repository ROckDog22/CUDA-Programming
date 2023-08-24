#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define N 3000
#define BLOCK_SIZE 32
int matrix[N*N];


__managed__ int a[N][N];
__managed__ int cpu_out[N][N];
__managed__ int gpu_out[N][N];

__global__ void gpu_ken(int a[N][N], int out[N][N]){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x <N && y<N){
        if (a[y][x] > 100){
            out[y][x] = 0;
        }else{
            out[y][x] = a[y][x];
        }
    }

}

void cpu_ken(int a[N][N], int out[N][N]){
    for(int i=0;i<N; ++i){
        for(int j=0; j<N; ++j){
            if(a[i][j]>100){
                out[i][j]=0;
            }else{
                out[i][j] = a[i][j];
            }
        }
    }
}


int main(){
    cudaEvent_t start, gpu_stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&gpu_stop));

    for(int i=0;i<N; ++i){
        for(int j=0; j<N; ++j){
            a[i][j] = rand()%1024;
        }
    }

    cpu_ken(a, cpu_out);
    
    cudaEventRecord(start);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

    gpu_ken<<<dimGrid, dimBlock>>>(a, gpu_out);

    cudaDeviceSynchronize();
    cudaEventRecord(gpu_stop); 
    cudaEventSynchronize(gpu_stop);
    float t;
    CHECK(cudaEventElapsedTime(&t, start, gpu_stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(gpu_stop));
    bool error = false;
    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            if(fabs(cpu_out[i][j] - gpu_out[i][j])>(1e-10)){
                error = true;
            }
        }
    }

    printf(error?"error":"pass");
    return 0;   

}