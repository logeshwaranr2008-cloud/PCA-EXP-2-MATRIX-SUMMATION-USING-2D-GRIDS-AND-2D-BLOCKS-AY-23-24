# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM:</h3>
<h3>ENTER YOUR NAME: LOGESHWARAN R</h3>
<h3>ENTER YOUR REGISTER NO: 212225040205</h3>
<h3>EX. NO: 02</h3>
<h3>DATE 02.11.2026</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%writefile matrix_add.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA error checking
#define CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Initialize matrix with random integers
void initialData(int *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = rand() % 100;
    }
}

// CPU matrix addition
void sumMatrixOnHost(int *A, int *B, int *C, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// GPU matrix addition using 2D grid and 2D blocks
__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C,
                                 int rows, int cols)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < cols && iy < rows)
    {
        int index = iy * cols + ix;
        C[index] = A[index] + B[index];
    }
}

// Compare CPU and GPU results
void checkResult(int *hostRef, int *gpuRef, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            printf("Arrays do not match at index %d\n", i);
            printf("CPU = %d, GPU = %d\n", hostRef[i], gpuRef[i]);
            return;
        }
    }

    printf("Arrays match successfully!\n");
}

int main()
{
    printf("2D Matrix Addition using CUDA\n\n");

    // Matrix dimensions
    int rows = 2048;
    int cols = 2048;
    int size = rows * cols;

    size_t nBytes = size * sizeof(int);

    printf("Matrix size: %d x %d\n\n", rows, cols);

    // Allocate host memory
    int *h_A = (int *)malloc(nBytes);
    int *h_B = (int *)malloc(nBytes);
    int *h_C = (int *)malloc(nBytes);
    int *h_D = (int *)malloc(nBytes);

    if (!h_A || !h_B || !h_C || !h_D)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize input matrices
    srand(1234);
    initialData(h_A, size);
    initialData(h_B, size);

    // CPU computation
    double cpuStart = (double)clock() / CLOCKS_PER_SEC;

    sumMatrixOnHost(h_A, h_B, h_C, rows, cols);

    double cpuEnd = (double)clock() / CLOCKS_PER_SEC;

    printf("CPU execution time: %f seconds\n\n", cpuEnd - cpuStart);

    // Allocate GPU memory
    int *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // Copy input matrices to GPU
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Different block configurations
    dim3 blockSizes[] = {
        dim3(8, 8),
        dim3(16, 16),
        dim3(32, 8),
        dim3(32, 16),
        dim3(32, 32)
    };

    int numConfigs = 5;
    float bestTime = 999999.0f;
    int bestConfig = 0;

    // CUDA events for measuring GPU time
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    printf("Block Configuration Comparison\n");
    printf("------------------------------------------\n");

    // Test each block configuration
    for (int i = 0; i < numConfigs; i++)
    {
        dim3 block = blockSizes[i];

        dim3 grid(
            (cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y
        );

        // Warm-up launch
        sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Measure GPU execution time
        CHECK(cudaEventRecord(start));

        for (int j = 0; j < 100; j++)
        {
            sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
        }

        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;

        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        float avgTime = milliseconds / 100.0f;

        printf("Block (%2d, %2d) | Grid (%4d, %4d) | Time: %f ms\n",
               block.x, block.y, grid.x, grid.y, avgTime);

        if (avgTime < bestTime)
        {
            bestTime = avgTime;
            bestConfig = i;
        }
    }

    // Run the best configuration for final result
    dim3 bestBlock = blockSizes[bestConfig];

    dim3 bestGrid(
        (cols + bestBlock.x - 1) / bestBlock.x,
        (rows + bestBlock.y - 1) / bestBlock.y
    );

    sumMatrixOnGPU2D<<<bestGrid, bestBlock>>>(d_A, d_B, d_C, rows, cols);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy result back to CPU
    CHECK(cudaMemcpy(h_D, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify results
    printf("\nResult Verification:\n");
    checkResult(h_C, h_D, size);

    // Display best configuration
    printf("\nBest Execution Configuration:\n");
    printf("Block size: (%d, %d)\n", bestBlock.x, bestBlock.y);
    printf("Grid size: (%d, %d)\n", bestGrid.x, bestGrid.y);
    printf("Average GPU time: %f ms\n", bestTime);

    // Free CUDA resources
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    // Reset GPU
    CHECK(cudaDeviceReset());

    return 0;
}
```
## OUTPUT:

<img width="662" height="562" alt="WhatsApp Image 2026-11-02 at 8 57 54 AM" src="https://github.com/user-attachments/assets/34f9724c-0ceb-4450-851e-ee94cbeac12c" />


## RESULT:
The host took _________ seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in ________ seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
