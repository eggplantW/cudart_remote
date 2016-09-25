#ifndef CUDART_REMOTE_H_
#define CUDART_REMOTE_H_

#include <cuda_runtime_api.h>
extern "C" {

extern __host__ void CUDARTAPI cudaRemoteInit(int *argc, char ***argv);
extern __host__ void CUDARTAPI cudaRemoteFinalize();

}

#endif /* CUDART_REMOTE_H_ */
