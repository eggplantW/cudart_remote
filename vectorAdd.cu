/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <pthread.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
void* add(void *id)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int *x = (int *)id;
    cudaSetDevice(*x);
		printf("x = %d\n",*x);
    // Print the vector length to be used, and compute its size
    int numElements = 1024 * 1024 * 256;
   	size_t size = numElements * sizeof(float);
		printf("pid = %d,numElements = %d,size = %u\n",pthread_self(),numElements,size);


    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);
    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);
		//printf("pid = %x,addrA = %lx,addrB = %lx,addrC = %lx\n",pthread_self(),h_A,h_B,h_C);
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
		h_A[i] = i;
		h_B[i] = i - 1;
    }
    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
		
    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		printf("pid = %d,memcpyA\n",pthread_self());

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (%d)!\n", err);
        exit(EXIT_FAILURE);
    }
		
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

			printf("pid = %d,memcpyB\n",pthread_self());

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
		cudaStreamSynchronize(0);
	err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector in device memory to the host result vector
    // in host memory.

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		//cudaStreamSynchronize(0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
	printf("pid = %d,memcpyC\n",pthread_self());
		int flag = 0;
	//	printf("local addr %d %d %d\n",d_A,d_B,d_C);
		// Verify that the result vector is correct
		for (int i = 0; i < numElements; ++i)
    {

        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            printf("Result verification failed at element %d %u!\n", i,pthread_self());
            flag = 1;
						//exit(EXIT_FAILURE);
						break;
        }
    }
		printf("%.3f+%.3f = %.3f\n",h_A[10],h_B[10],h_C[10]);
		if(flag)
			printf("-------------------------------------------Test Fail %u-------------------------------------------------------\n",pthread_self());
		else
    	printf("-------------------------------------------Test PASSED %u-----------------------------------------------------\n",pthread_self());
		/*
		cudaSetDevice((*x) * 2 + 1);
		float *d_A1 = NULL;
    err = cudaMalloc((void **)&d_A1, size);
		float *d_A2 = NULL;
		err = cudaMalloc((void **)&d_A2,size);
		float *d_A3 = NULL;
		float *h_A1 = (float *)malloc(size);
		err = cudaMalloc((void **)&d_A3,size);
		err = cudaMemcpy(d_A1, d_A, size, cudaMemcpyDeviceToDevice);
		err = cudaMemcpy(d_A2, d_B, size, cudaMemcpyDeviceToDevice);
	 	threadsPerBlock = 256;
    blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_A2, d_A3, numElements);
		cudaStreamSynchronize(0);

		err = cudaMemcpy(h_A1, d_A3, size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < numElements; ++i)
    {

        if (fabs(h_A[i] + h_B[i] - h_A1[i]) > 1e-5)
        {
            printf("Result verification failed at element %d %u!\n", i,pthread_self());
            flag = 1;
						//exit(EXIT_FAILURE);
						break;
        }
    }
		if(flag)
			printf("-------------------------------------------Test Fail 1 %u-------------------------------------------------------\n",pthread_self());
		else
    	printf("-------------------------------------------Test PASSED 1 %u-----------------------------------------------------\n",pthread_self());
		cudaFree(d_A1);
		cudaFree(d_A2);
		cudaFree(d_A3);
		cudaSetDevice((*x) * 2);
		*/
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits


	
    return;
}
int main()
{
	pthread_t pid[1000];
	int id[32];
	int i,j;
	for(i = 0;i < 32;i ++)
		id[i] = i;
	for(i = 0;i < 24;i ++)
		pthread_create(&pid[i],NULL,add,(void *)&id[i % 32]);
	
	for(i = 0;i < 24;i ++)
	{
		//pthread_create(&pid[i],NULL,add,(void *)&id[i % 8]);
		pthread_join(pid[i],NULL);
		printf("thread %d fini\n",i);
	}
	
	return 0;
}
