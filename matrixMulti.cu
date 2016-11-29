/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
// CUDA runtime
#include "cudart_remote.h"

// Helper functions and utilities to work with CUDA
#include "Functions.h"

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int block_size, dim3 &dimsA, dim3 &dimsB)
{
	cudaSetDevice(0);
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);
    constantInit(h_C, dimsC.x * dimsC.y, 0);

    cuda_error( cudaMalloc((void **) &d_A, mem_size_A) );
    cuda_error( cudaMalloc((void **) &d_B, mem_size_B) );
    cuda_error( cudaMalloc((void **) &d_C, mem_size_C) );

    // copy host memory to device
    cuda_error( cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    cuda_error( cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16)
    {
        matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    // Copy result from device to host
    cuda_error( cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}


int matrixMultiplyRange(int block_size, size_t maxDim)
{
	cudaSetDevice(0);
    // Allocate host memory for matrices A and B
	size_t element_count = maxDim * maxDim;
	size_t mem_size = element_count * sizeof(float);
    float *h_A = (float *)malloc(mem_size);
    float *h_B = (float *)malloc(mem_size);
    float *h_C = (float *)malloc(mem_size);
    if(h_A == NULL || h_B == NULL || h_C == NULL) {
    	printf("host memory allocate failed!\n");
    	return EXIT_FAILURE;
    }
    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, element_count, 1.0f);
    constantInit(h_B, element_count, valB);
    constantInit(h_C, element_count, 0);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cuda_error( cudaMalloc((void **) &d_A, mem_size) );
    cuda_error( cudaMalloc((void **) &d_B, mem_size) );
    cuda_error( cudaMalloc((void **) &d_C, mem_size) );
    printf("   dim\t  time/s\n");
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(maxDim / block_size, maxDim / block_size);
	cuda_error( cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice) );
	cuda_error( cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice) );
	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16)
	{
		matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, maxDim, maxDim);
	}
	else
	{
		matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, maxDim, maxDim);
	}
	// Copy result from device to host
	cuda_error( cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost) );

    timeval start_time, htodend_time, dtohstart_time, end_time;
    size_t _size;
    for(size_t d = 1024; d <= maxDim; d+=1024) {
    	_size = d * d * sizeof(float);
    	grid.x = d / block_size;
    	grid.y = d / block_size;
    	gettimeofday(&start_time, 0);
		// copy host memory to device
		//cuda_error( cudaMemcpy(d_A, h_A, _size, cudaMemcpyHostToDevice) );
		//cuda_error( cudaMemcpy(d_B, h_B, _size, cudaMemcpyHostToDevice) );
		// Performs warmup operation using matrixMul CUDA kernel
		//gettimeofday(&htodend_time, 0);
		if (block_size == 16)
		{
			matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, d, d);
		}
		else
		{
			matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, d, d);
		}
		//gettimeofday(&dtohstart_time, 0);
		// Copy result from device to host
		//cuda_error( cudaMemcpy(h_C, d_C, _size, cudaMemcpyDeviceToHost) );
		cuda_error( cudaStreamSynchronize(0) );
		gettimeofday(&end_time, 0);
		//double timehtod = (double)1000000*(htodend_time.tv_sec - start_time.tv_sec) + (htodend_time.tv_usec - start_time.tv_usec);
		//double timedtoh = (double)1000000*(end_time.tv_sec - dtohstart_time.tv_sec) + (end_time.tv_usec - dtohstart_time.tv_usec);
		double timeuse = (double)1000000*(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
		printf("%6d\t%.6f\n", d, timeuse / 1000000);
    }



    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}

int matrixMultiplyStreamRange(int block_size, size_t maxDim, unsigned int streamNum)
{
	cudaSetDevice(0);
    // Allocate host memory for matrices A and B
	maxDim = (maxDim > 8192)? 8192 : maxDim;
	size_t element_count = maxDim * maxDim;
	size_t mem_size = element_count * sizeof(float);
	streamNum = (streamNum > 5)? 5 : streamNum;

	float** h_A = (float**)malloc(sizeof(float*) * streamNum * 3);
	float** h_B = h_A + streamNum;
	float** h_C = h_B + streamNum;
    if(h_A == NULL) {
    	printf("host memory allocate failed!\n");
    	return EXIT_FAILURE;
    }
	for(int i = 0; i < streamNum; i++) {
		h_A[i] = (float*)malloc(mem_size);
		h_B[i] = (float*)malloc(mem_size);
		h_C[i] = (float*)malloc(mem_size);
		if(h_A[i] == NULL || h_B[i] == NULL || h_C[i] == NULL) {
	    	printf("host memory allocate failed!\n");
	    	return EXIT_FAILURE;
		}
	}

    // Initialize host memory
    const float valB = 0.01f;
    for(int i = 0; i < streamNum; i++) {
		constantInit(h_A[i], element_count, 1.0f);
		constantInit(h_B[i], element_count, valB);
		constantInit(h_C[i], element_count, 0);
    }

    // Allocate device memory
	float** d_A = (float**)malloc(sizeof(float*) * streamNum * 3);
	float** d_B = d_A + streamNum;
	float** d_C = d_B + streamNum;
    if(d_A == NULL) {
    	printf("host memory allocate failed!\n");
    	return EXIT_FAILURE;
    }
    for(int i = 0; i < streamNum; i++) {
    	cuda_error( cudaMalloc((void **) d_A + i, mem_size) );
    	cuda_error( cudaMalloc((void **) d_B + i, mem_size) );
    	cuda_error( cudaMalloc((void **) d_C + i, mem_size) );
    }
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * streamNum);
    streams[0] = 0;
    for(int i = 1; i < streamNum; i++)
    	cuda_error( cudaStreamCreate(streams + i) );
    printf("   dim\t  time/s\n");
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(maxDim / block_size, maxDim / block_size);
	for(int i = 0; i < streamNum; i++) {
		cuda_error( cudaMemcpyAsync(d_A[i], h_A[i], mem_size, cudaMemcpyHostToDevice, streams[i]) );
		cuda_error( cudaMemcpyAsync(d_B[i], h_B[i], mem_size, cudaMemcpyHostToDevice, streams[i]) );
		// Performs warmup operation using matrixMul CUDA kernel
		if (block_size == 16)
		{
			matrixMulCUDA<16><<< grid, threads, 0, streams[i] >>>(d_C[i], d_A[i], d_B[i], maxDim, maxDim);
		}
		else
		{
			matrixMulCUDA<32><<< grid, threads, 0, streams[i] >>>(d_C[i], d_A[i], d_B[i], maxDim, maxDim);
		}
		// Copy result from device to host
		cuda_error( cudaMemcpyAsync(h_C[i], d_C[i], mem_size, cudaMemcpyDeviceToHost, streams[i]) );
	}
	cuda_error( cudaDeviceSynchronize() );
    timeval start_time, end_time;
    size_t _size;
    for(size_t d = 64; d <= maxDim; d+=64) {
    	_size = d * d * sizeof(float);
    	grid.x = d / block_size;
    	grid.y = d / block_size;
    	gettimeofday(&start_time, 0);
		// copy host memory to device
    	for(int i = 0; i < streamNum; i++) {
			cuda_error( cudaMemcpyAsync(d_A[i], h_A[i], _size, cudaMemcpyHostToDevice, streams[i]) );
			cuda_error( cudaMemcpyAsync(d_B[i], h_B[i], _size, cudaMemcpyHostToDevice, streams[i]) );
			// Performs warmup operation using matrixMul CUDA kernel
			if (block_size == 16)
			{
				matrixMulCUDA<16><<< grid, threads, 0, streams[i] >>>(d_C[i], d_A[i], d_B[i], d, d);
			}
			else
			{
				matrixMulCUDA<32><<< grid, threads, 0, streams[i] >>>(d_C[i], d_A[i], d_B[i], d, d);
			}
			// Copy result from device to host
			cuda_error( cudaMemcpyAsync(h_C[i], d_C[i], _size, cudaMemcpyDeviceToHost, streams[i]) );
    	}
    	cuda_error( cudaDeviceSynchronize() );
		gettimeofday(&end_time, 0);
		double timeuse = (double)1000000*(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
		printf("%6d\t%.6f\n", d, timeuse / 1000000 / streamNum);
    }

    for(int i = 1; i < streamNum; i++)
    	cuda_error( cudaStreamDestroy(streams[i]) );

    // Clean up memory
    for(int i = 0; i < streamNum; i++) {
    	free(h_A[i]);
    	free(h_B[i]);
    	free(h_C[i]);
    	cudaFree(d_A[i]);
    	cudaFree(d_B[i]);
    	cudaFree(d_C[i]);
    }
    free(h_A);
    free(d_A);

    return EXIT_SUCCESS;
}

int matrixMultiplyMultiGPU(int block_size, size_t maxDim) {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("cnt: %d\n", deviceCount);
	//deviceCount = 1;
    // Allocate host memory for matrices A and B
	size_t element_count = maxDim * maxDim;
	size_t mem_size = element_count * sizeof(float);

	float** h_A = (float**)malloc(sizeof(float*) * deviceCount * 3);
	float** h_B = h_A + deviceCount;
	float** h_C = h_B + deviceCount;
    if(h_A == NULL) {
    	printf("host memory allocate failed!\n");
    	return EXIT_FAILURE;
    }
	for(int i = 0; i < deviceCount; i++) {
		h_A[i] = (float*)malloc(mem_size);
		h_B[i] = (float*)malloc(mem_size);
		h_C[i] = (float*)malloc(mem_size);
		if(h_A[i] == NULL || h_B[i] == NULL || h_C[i] == NULL) {
	    	printf("host memory allocate failed!\n");
	    	return EXIT_FAILURE;
		}
	}

    // Initialize host memory
    const float valB = 0.01f;
    for(int i = 0; i < deviceCount; i++) {
		constantInit(h_A[i], element_count, 1.0f);
		constantInit(h_B[i], element_count, valB);
		constantInit(h_C[i], element_count, 0);
    }

    // Allocate device memory
	float** d_A = (float**)malloc(sizeof(float*) * deviceCount * 3);
	float** d_B = d_A + deviceCount;
	float** d_C = d_B + deviceCount;
    if(d_A == NULL) {
    	printf("host memory allocate failed!\n");
    	return EXIT_FAILURE;
    }
    for(int i = 0; i < deviceCount; i++) {
    	cuda_error( cudaSetDevice(i) );
    	cuda_error( cudaMalloc((void **) d_A + i, mem_size) );
    	cuda_error( cudaMalloc((void **) d_B + i, mem_size) );
    	cuda_error( cudaMalloc((void **) d_C + i, mem_size) );
    }

    printf("   dim\t  time/s\n");
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(maxDim / block_size, maxDim / block_size);
    size_t testSize = 1024*1024*sizeof(float);
	for(int i = 0; i < deviceCount; i++) {
		cuda_error( cudaSetDevice(i) );
		cuda_error( cudaMemcpyAsync(d_A[i], h_A[i], testSize, cudaMemcpyHostToDevice, 0) );
		cuda_error( cudaMemcpyAsync(d_B[i], h_B[i], testSize, cudaMemcpyHostToDevice, 0) );
		// Performs warmup operation using matrixMul CUDA kernel
		if (block_size == 16)
		{
			matrixMulCUDA<16><<< grid, threads, 0, 0 >>>(d_C[i], d_A[i], d_B[i], 1024, 1024);
		}
		else
		{
			matrixMulCUDA<32><<< grid, threads, 0, 0 >>>(d_C[i], d_A[i], d_B[i], 1024, 1024);
		}
		// Copy result from device to host
		cuda_error( cudaMemcpyAsync(h_C[i], d_C[i], testSize, cudaMemcpyDeviceToHost, 0) );
	}
	printf("1\n");
	for(int i = 0; i < deviceCount; i++) {
		cuda_error( cudaSetDevice(i) );
		cuda_error( cudaStreamSynchronize(0) );
	}
	printf("2\n");
    timeval start_time, end_time;
    size_t _size;
    for(size_t d = 1024; d <= maxDim; d+=1024) {
    	_size = d * d * sizeof(float);
    	grid.x = d / block_size;
    	grid.y = d / block_size;
    	gettimeofday(&start_time, 0);
		// copy host memory to device
    	for(int i = 0; i < deviceCount; i++) {
    		cuda_error( cudaSetDevice(i) );
			cuda_error( cudaMemcpyAsync(d_A[i], h_A[i], _size, cudaMemcpyHostToDevice, 0) );
			cuda_error( cudaMemcpyAsync(d_B[i], h_B[i], _size, cudaMemcpyHostToDevice, 0) );
			// Performs warmup operation using matrixMul CUDA kernel
			if (block_size == 16)
			{
				matrixMulCUDA<16><<< grid, threads, 0, 0 >>>(d_C[i], d_A[i], d_B[i], d, d);
			}
			else
			{
				matrixMulCUDA<32><<< grid, threads, 0, 0 >>>(d_C[i], d_A[i], d_B[i], d, d);
			}
			// Copy result from device to host
			cuda_error( cudaMemcpyAsync(h_C[i], d_C[i], _size, cudaMemcpyDeviceToHost, 0) );
    	}
    	for(int i = 0; i < deviceCount; i++) {
    		cuda_error( cudaSetDevice(i) );
    		cuda_error( cudaDeviceSynchronize() );
    	}
		gettimeofday(&end_time, 0);
		double timeuse = (double)1000000*(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
		printf("%6d\t%.6f\n", d, timeuse / 1000000);
    }

    // Clean up memory
    for(int i = 0; i < deviceCount; i++) {
    	free(h_A[i]);
    	free(h_B[i]);
    	free(h_C[i]);
    	cudaSetDevice(i);
    	cudaFree(d_A[i]);
    	cudaFree(d_B[i]);
    	cudaFree(d_C[i]);
    }
    free(h_A);
    free(d_A);

    return EXIT_SUCCESS;
}
