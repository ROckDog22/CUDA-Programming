#include <stdio.h>
#include <math.h>
#include "error.cuh"
#define BLOCK_SIZE 2
#define M 5
#define N 5

__managed__ int matrix[N*M];
__managed__ int cpu_matrix[M*N];
__managed__ int gpu_matrix[M*N];

__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N])
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if( x < M && y < N)
    {
        out[x][y] = in[y][x];
    }
}

__global__ void gpu_shared_matrix_transpose(int *a, int *b, int m, int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int s_a[(BLOCK_SIZE+1)* BLOCK_SIZE];

    if(x<n && y<m){
        s_a[threadIdx.x * BLOCK_SIZE + threadIdx.y] = a[x*m + y];
    }

    __syncthreads();

    int x1 = blockIdx.x * blockDim.x + threadIdx.y;
    int y1 = blockIdx.y * blockDim.y + threadIdx.x;
    if(x1<n && y1<m){
        b[y1*n + x1] = s_a[threadIdx.y*BLOCK_SIZE + threadIdx.x];
    }

}

void matrix_transpose(int *a, int *b, int m, int n){
    for(int i=0; i<m; ++i){
        for(int j=0; j<n; ++j){
            b[i*n + j] = a[j*m + i];
        }
    }
}

int main(){
    cudaEvent_t start, mt_cpu, mt_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&mt_cpu));
    CHECK(cudaEventCreate(&mt_gpu));

    for(int i = 0; i<N; ++i){
        for(int j=0; j<M; ++j){
            matrix[i*N + j] = rand() % 1024;
        }
    }

    CHECK(cudaEventRecord(start));
    matrix_transpose(matrix, cpu_matrix, M, N);

    CHECK(cudaEventRecord(mt_cpu));
    CHECK(cudaEventSynchronize(mt_cpu));

    dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    for(int i=0; i<20; ++i){
        gpu_shared_matrix_transpose<<<dimGrid, dimBlock>>>(matrix, gpu_matrix, M, N);
    }

    CHECK(cudaEventRecord(mt_gpu));
    CHECK(cudaEventSynchronize(mt_gpu));

    bool errors= false;
    float cpu_time, gpu_time;
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if(fabs(cpu_matrix[i*N + j] - gpu_matrix[i*N + j])>(1.0e-10)){
                errors = true;
                // printf("cpu_value: %d, gpu_value: %d\n", cpu_matrix[i*N + j], gpu_matrix[i*N + j]);
            }
        }
    }
    printf("--------------------------")
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            printf("%d  ", cpu_matrix[i*N + j]);
        }
        printf("\n");
    }
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            printf("%d  ", gpu_matrix[i*N + j]);
        }
        printf("\n");
    }
    if(errors){
        printf(errors? "error":"pass");
    }

    cudaEventElapsedTime(&cpu_time, start, mt_cpu);
    cudaEventElapsedTime(&gpu_time, mt_cpu, mt_gpu);

    return 0;


}   

