/*
 * Functions.h
 *
 *  Created on: 2015-6-17
 *      Author: makai
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <iostream>
#include <string.h>

#define cuda_error_drv(error) _cuda_error_drv(error, __FILE__, __LINE__)
#define mpi_error(error) _mpi_error(error, __FILE__, __LINE__)
#define cuda_error(error) _cuda_error(error, __FILE__, __LINE__)
#define print_status(status) _print_status(status, __FILE__, __LINE__)
#define syscall_error(error) _syscall_error(error, __FILE__, __LINE__)


inline void _cuda_error_drv(CUresult err, const char *file, int line) {
	if(err == CUDA_SUCCESS)
		return;
	const char *errString;
	cuGetErrorString(err, &errString);
	std::cerr << "cuda error: " << errString << " at " << file << ":" << line << std::endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}

inline void _mpi_error(int err, const char *file, int line) {
	if(err == MPI_SUCCESS)
		return;
	char errString[256];
	int errLen = 0;
	MPI_Error_string(err, errString, &errLen);
	std::cerr << "mpi error: " << errString << " at " << file << ":" << line << std::endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}

inline void _cuda_error(cudaError_t err, const char *file, int line) {
	if(err == cudaSuccess)
		return;
	std::cerr << "cuda error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
	//printf("cuda error %s at %s:%d",cudaGetErrorString(err),file,line);
	MPI_Abort(MPI_COMM_WORLD, -1);
}

inline void _print_status(MPI_Status& status, const char *file, int line) {
	int count;
	MPI_Get_count(&status, MPI_BYTE, &count);
	std::cout << "Src: " << status.MPI_SOURCE << ", Tag: " << status.MPI_TAG << ", Count: " << count << " at " << file << ":" << line << std::endl;
}

inline void _syscall_error(int err, const char* file, int line) {
	if(err == 0)
		return;
	std::cerr << "error string: " << strerror(err) << " at " << file << ":" << line << std::endl;
	MPI_Abort(MPI_COMM_WORLD, -1);
}

#endif /* FUNCTIONS_H_ */
