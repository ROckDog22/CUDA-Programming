#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main(void){
    cudaDeviceProp prop;
    int whichDevice;
    CHECK(cudaGetDevice(&whichDevice));
    CHECK(cudaGetDeviceProperties(&prop, whichDevice));
    if(!porp.deviceOverlap){
        // 是否支持同时进行数据传输和核函数调用
        printf("no")
    }
    
}