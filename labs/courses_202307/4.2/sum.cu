#include <math.h>
#include <stdio.h>

#include "error.cuh"

#define N 1000000
#define BLOCK_SIZE 256
#define GRID_SIZE 256

__managed__ int source[N];
__managed__ int gpu_result[1] = {0};
__managed__ int gpu_1_pass_result[GRID_SIZE] = {0};
__managed__ int gpu_2_pass_result[1] = {0};
__managed__ int gpu_atomic[1] = {0};
__managed__ int gpu_warp_shuffle_result[1] = {0};


__global__ void sum_gpu_naive(int *a, int count, int *out){
    for(int i=blockIdx.x * blockDim.x + threadIdx.x ; i<count; i+=blockDim.x*gridDim.x){
        atomicAdd(out, a[i]);
    }
}

__global__ void _shared_2pass_sum_gpu(int *in, int count, int *out)
{
    __shared__ int ken[BLOCK_SIZE];
    int sum_1 = 0;
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<count; i+=blockDim.x * gridDim.x){
        sum_1 += in[i];
    }
    ken[threadIdx.x] = sum_1;
    __syncthreads();
    
    for(int thread_stride = BLOCK_SIZE/2; thread_stride>=1; thread_stride/=2){
        if(threadIdx.x < thread_stride){
            ken[threadIdx.x] = ken[threadIdx.x] + ken[threadIdx.x+thread_stride];
        }

        __syncthreads();
        
    }

    if(blockIdx.x * blockDim.x < count){
        if(threadIdx.x == 0){
            out[blockIdx.x] = ken[0];
        }
    }

}

__global__ void _shared_atomic_sum_gpu(int *in, int count, int *out)
{
    __shared__ int ken[BLOCK_SIZE];
    //grid_loop
    int shared_tmp=0;
    for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += blockDim.x * gridDim.x)
    {
        shared_tmp +=in[idx];
    }
    ken[threadIdx.x] = shared_tmp;
    __syncthreads();

    for(int total_threads = BLOCK_SIZE/2; total_threads>=1; total_threads/=2)
    {
        if(threadIdx.x < total_threads)
        {
            ken[threadIdx.x] = ken[threadIdx.x] + ken[threadIdx.x + total_threads]; 
        }
        __syncthreads();
    }
    // block_sum -> share memory[0]
    if(blockIdx.x * blockDim.x < count)
    {
        if(threadIdx.x == 0)
        {
            atomicAdd(out, ken[0]);
        }
    }
}

__global__ void _shared_atomic_shuffle_sum_gpu(int *in, int count, int *out)
{
    __shared__ int ken[BLOCK_SIZE];
    //grid_loop
    int shared_tmp=0;
    for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += blockDim.x * gridDim.x)
    {
        shared_tmp +=in[idx];
    }
    ken[threadIdx.x] = shared_tmp;
    __syncthreads();

    for(int total_threads = BLOCK_SIZE/2; total_threads>=32; total_threads/=2)
    {
        if(threadIdx.x < total_threads)
        {
            ken[threadIdx.x] = ken[threadIdx.x] + ken[threadIdx.x + total_threads]; 
        }
        __syncthreads();
    }
    int val = ken[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // block_sum -> share memory[0]
    if(blockIdx.x * blockDim.x < count)
    {
        if(threadIdx.x == 0)
        {
            atomicAdd(out, val);
        }
    }
}

int main(){
    int cpu_result = 0;

    printf("Init source N\n");
    
    for(int i=0; i<N; ++i){
        source[i] = rand()%10;
    }

    cudaEvent_t start, stop_cpu, stop_gpu_naive, stop_gpu_2pass, stop_gpu_atomic, stop_gpu_shuffle;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventCreate(&stop_gpu_naive));
    CHECK(cudaEventCreate(&stop_gpu_2pass));
    CHECK(cudaEventCreate(&stop_gpu_atomic));
    CHECK(cudaEventCreate(&stop_gpu_shuffle));
    // CHECK(cudaEventCreate());

    cudaEventRecord(start);
    
    for(int i=0; i<20; ++i){
        gpu_result[0] = 0;
        sum_gpu_naive<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, gpu_result);
        CHECK(cudaDeviceSynchronize());
    }

    cudaEventRecord(stop_gpu_naive);
    cudaEventSynchronize(stop_gpu_naive);

    for(int i=0; i<N;++i){
        cpu_result+=source[i];
    }
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    _shared_2pass_sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, gpu_1_pass_result);
    _shared_2pass_sum_gpu<<<1, BLOCK_SIZE>>>(gpu_1_pass_result, GRID_SIZE, gpu_2_pass_result);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_gpu_2pass);
    cudaEventSynchronize(stop_gpu_2pass);

    for(int i=0; i<20; ++i){
        gpu_atomic[0] = 0;
        _shared_atomic_sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, gpu_atomic);
        CHECK(cudaDeviceSynchronize());
    }

    cudaEventRecord(stop_gpu_atomic);
    cudaEventSynchronize(stop_gpu_atomic);


    for(int i = 0; i<20; i++)
    {
        gpu_warp_shuffle_result[0] = 0;
        _shared_atomic_shuffle_sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, gpu_warp_shuffle_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu_shuffle);
    cudaEventSynchronize(stop_gpu_shuffle);

    float time_cpu, time_gpu_naive, time_gpu_2pass, time_gpu_atomic, time_gpu_shuffle;

    cudaEventElapsedTime(&time_gpu_naive, start, stop_gpu_naive);
    cudaEventElapsedTime(&time_cpu, stop_gpu_naive, stop_cpu);
    cudaEventElapsedTime(&time_gpu_2pass, stop_cpu, stop_gpu_2pass);
    cudaEventElapsedTime(&time_gpu_atomic, stop_gpu_2pass, stop_gpu_atomic);
    cudaEventElapsedTime(&time_gpu_shuffle, stop_gpu_atomic, stop_gpu_shuffle);
    printf("cpu_time: %.2f result %d\n", time_cpu, cpu_result);
    printf("sum_gpu_naive: %.2f  result: %d %s\n", time_gpu_naive/20, gpu_result[0], cpu_result == gpu_result[0]?"pass": "error");
    printf("sum_gpu_2pass: %.2f  result: %d %s\n", time_gpu_2pass, gpu_2_pass_result[0], (cpu_result == gpu_2_pass_result[0]? "pass" : "error"));
    printf("sum_gpu_atomic: %.2f  result: %d %s\n", time_gpu_atomic/20, gpu_atomic[0], (cpu_result == gpu_atomic[0]? "pass" : "error"));
    printf("sum_gpu_shuffle_atomic: %.2f  result: %d %s\n", time_gpu_atomic/20, gpu_atomic[0], (cpu_result == gpu_atomic[0]? "pass" : "error"));
    return 0;
}