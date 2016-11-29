/*
 * Functions.cpp
 *
 *  Created on: 2015-6-17
 *      Author: makai
 */


#include "Functions.h"

using namespace std;

inline void _cuda_error_drv(CUresult err, const char *file, int line) {
	if(err == CUDA_SUCCESS)
		return;
	const char *errString;
	cuGetErrorString(err, &errString);
	cout << "cuda error: " << errString << " at " << file << ":" << line << endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}

inline void _mpi_error(int err, const char *file, int line) {
	if(err == MPI_SUCCESS)
		return;
	char errString[256];
	int errLen = 0;
	MPI_Error_string(err, errString, &errLen);
	cout << "mpi error: " << errString << " at " << file << ":" << line << endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}

inline void _cuda_error(cudaError_t err, const char *file, int line) {
	if(err == cudaSuccess)
		return;
	cout << "cuda error: " << cudaGetErrorString(err) << " at " << file << ":" << line << endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}
