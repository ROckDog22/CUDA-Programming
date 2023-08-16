#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 16
// 为什么不要用索引呢，因为写是没有缓存的每次都要访问global memory 导致相比于
// 寄存器和local memeory的变量就速度慢了很多
__global__ void gpu_matrix_mul1(int* A, int *B, int *C, int m, int n, int k){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row <m && col <k){
        int index = row* k + col;
        C[index] = 0;
        for(int i = 0; i<n; ++i){
            C[index] += A[row * n + i] * B[i*k+col];
        }
    }
}

__global__ void gpu_matrix_mul2(int* A, int *B, int *C, int m, int n, int k){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;
    if(row <m && col <k){
        for(int i = 0; i<n; ++i){
            sum += A[row * n + i] * B[i*k+col];
        }
        C[row*k + col] = sum;
    }
}

void cpu_matrix_mul1(int * A, int *B, int *C, int m, int n, int k){
    for(int i = 0; i<m; ++i){
        for(int j = 0; j<k; ++j){
            C[i*k+j] = 0;
            for(int q = 0; q<k; ++q){
                C[i*k+j] += A[i*n + q]*B[q*k + j];
            }
        }
    }
}

void cpu_matrix_mul2(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, const char *arg[]){
    int m = 100;
    int n = 100;
    int k = 100;
    
    int *h_a = (int*) malloc(sizeof(int) * m * n);
    int *h_b = (int*) malloc(sizeof(int) * n * k);
    int *h_c = (int*) malloc(sizeof(int) * m * k);
    int *h_cc = (int*) malloc(sizeof(int) * m * k);

    for(int i = 0; i<m; ++i){
        for(int j = 0; j<n; ++j){
            h_a[i*n + j] = rand() % 1024;
        }
    }

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<k; ++j){
            h_b[i*k + j] = rand() % 1024;
        }
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(int) * m * n);
    cudaMalloc((void **)&d_b, sizeof(int) * n * k);
    cudaMalloc((void **)&d_c, sizeof(int) * m * k);
    
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    
    unsigned int row  = (m + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int col = (k + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(col, row);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mul2<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    gpu_matrix_mul1<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cpu_matrix_mul2(h_a, h_b, h_cc, m, n, k);
    cpu_matrix_mul1(h_a, h_b, h_cc, m, n, k);
    int ok = 1;
    for(int i=0; i<m; ++i){
        for(int j=0; j<n; ++j){
            if(h_c[i*n+j] - h_cc[i*n+j] >=(1e-10)){
                ok = 0;
            }
        }
    }

    if(ok==0){
        printf("fail!!\n");
    }else{
        printf("pass!!\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_cc);
    return 0;
}

