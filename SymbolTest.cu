
#include <cuda_runtime_api.h>
#include <time.h>
#include <stdlib.h>
#include "Functions.h"

__device__ int SymbolArray1[32];
__constant__ int SymbolArray2[32];

__global__ void SymbolKernel() {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < 32)
		SymbolArray1[tid] += SymbolArray2[tid];
}

void SymbolTest() {
	cuda_error( cudaSetDevice(0) );
	int harray1[32];
	int harray2[32];
	int result[32];

	for(int i = 0; i < 32; i++) {
		harray1[i] = rand() % 10000;
		harray2[i] = rand() % 10000;
	}
	cuda_error( cudaMemcpyToSymbol(SymbolArray1, harray1, 32*sizeof(int)) );
	cuda_error( cudaMemcpyToSymbol(SymbolArray2, harray2, 32*sizeof(int)) );
	SymbolKernel<<<1, 32>>>();
	cuda_error( cudaMemcpyFromSymbol(result, SymbolArray1, 32*sizeof(int)) );
	for(int i = 0; i < 32; i++) {
		if(result[i] != harray1[i] + harray2[i]) {
			printf("error at: %d\n", i);
			return;
		}
	}
	printf("PASSED\n");
}
