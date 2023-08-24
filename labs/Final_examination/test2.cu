// 利用GPU找出一列包含1000000个元素的数组的**最大的10个值**.
#incldue <stdio.h>
#incldue <math.h>

#defint N 1000000
__managed__ int matrix[N];

void cpu_topk(int *input, int *output, int length, int k){
    for(int )
}



int main()
{
    printf("Init source data...........\n");
    for(int i=0; i<N; i++)
    {
        source[i] = rand();
    }

    printf("Complete init source data.....\n");
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);
    printf("GPU Run **************\n");
    for(int i =0; i<20; i++)
    {
        gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(source, _1_pass_result, N, topk);

        gpu_topk<<<1, BLOCK_SIZE>>>(_1_pass_result, gpu_result, topk * GRID_SIZE, topk);
        // gpu_topk<<<1, BLOCK_SIZE>>>(source, gpu_result, N, topk);

        cudaDeviceSynchronize();
    }
    printf("GPU Complete!!!\n");
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    int cpu_result[topk] ={0};
    printf("CPU RUN***************\n");
    // event 可以记录cpu上函数的执行事件
    cpu_topk(source, cpu_result, N, topk);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    printf("CPU Complete!!!!!");

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool error = false;
    for(int i =0; i<topk; i++)
    {
        printf("CPU top%d: %d; GPU top%d: %d;\n", i+1, cpu_result[i], i+1, gpu_result[i]);
        if(fabs(gpu_result[i] - cpu_result[i]) > 0)
        {
            error = true;
        }
    }
    printf("Result: %s\n", (error?"Error":"Pass"));
    printf("CPU time: %.2f; GPU time: %.2f\n", time_cpu, (time_gpu/20.0));
}




