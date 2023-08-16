#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void gpu_transpose1(int *A, int *B, int m){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < m && col <m){
        B[row*m + col] = A[col*m + row];
    }
}

void cpu_transpose1(int *A, int *B, int m){
    for(int i=0; i<m; ++i){
        for(int j=0; j<m;++j){
            B[i*m + j] = A[j*m+i];
        }
    }
}


__global__ void gpu_transpose(int *in,int *out, int width)
{ 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( y < width && x < width)
    {
        out[x * width + y] = in[y * width + x];
    }
} 

void cpu_transpose(int *in, int *out, int width)
{
    for(int y = 0; y < width; y++)
    {
        for(int x = 0; x < width; x++)
        {
            out[x * width + y] = in[y * width + x];
        }
    }
}
int main(int argc, const char* argv[]){
    int m = 1000;
    int *h_a = (int*)malloc(sizeof(int)*m*m);
    int *h_b = (int*)malloc(sizeof(int)*m*m);
    int *h_b_gpu = (int*)malloc(sizeof(int)*m*m);
    int* d_a, *d_b;

    for(int i =0; i<m; ++i){
        for(int j=0; j<m; ++j){
            h_a[i*m + j] = rand()%1024;
        }
    }
    cudaMalloc((void **)&d_a, sizeof(int)*m*m);
    cudaMalloc((void **)&d_b, sizeof(int)*m*m);

    cudaMemcpy(d_a, h_a, sizeof(int)*m*m, cudaMemcpyHostToDevice);
    
    unsigned int row = (m-1+BLOCK_SIZE) / BLOCK_SIZE;
    unsigned int col = (m-1+BLOCK_SIZE) / BLOCK_SIZE;
    
    dim3 dimGrid(col, row);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_transpose<<<dimGrid, dimBlock>>>(d_a, d_b, m);
    cpu_transpose(h_a, h_b, m);
    cudaMemcpy(h_b_gpu, d_b, sizeof(int)*m*m, cudaMemcpyDeviceToHost);

    int ok =1;
    for(int i = 0;i <m;++i){
        for(int j=0;j<m;++j){
            if(fabs(h_b[i*m+j]-h_b_gpu[i*m+j])>(1e-10)){
                ok=0;
            }
        }
    }
    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    gpu_transpose1<<<dimGrid, dimBlock>>>(d_a, d_b, m);
    cpu_transpose1(h_a, h_b, m);
    cudaMemcpy(h_b_gpu, d_b, sizeof(int)*m*m, cudaMemcpyDeviceToHost);

    ok =1;
    for(int i = 0;i <m;++i){
        for(int j=0;j<m;++j){
            if(fabs(h_b[i*m+j]-h_b_gpu[i*m+j])>(1e-10)){
                ok=0;
            }
        }
    }
    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(h_b_gpu);
    return 0;
}