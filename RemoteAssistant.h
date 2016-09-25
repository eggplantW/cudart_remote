#ifndef REMOTEASSISTANT_H_
#define REMOTEASSISTANT_H_

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Functions.h"
#include "Types.h"
#include "Defines.h"
#include "Stream.h"

#include <map>
#include <string>
#include <vector>

#include <pthread.h>
#include <semaphore.h>

#define TASK_QUE_SIZE 8192
#define TASK_BUF_SIZE 128

struct CudaLink_t {

	static const size_t JIT_BUF_SIZE = 8192;
	CUjit_option options[6];
	void* values[6];
	float walltime;
	char error_log[JIT_BUF_SIZE], info_log[JIT_BUF_SIZE];

	CUlinkState linkState;
	void* cubin;
	size_t cubinSize;

	CudaLink_t():cubin(NULL), cubinSize(0) {
		options[0] = CU_JIT_WALL_TIME;
		values[0] = (void*)&walltime;
		options[1] = CU_JIT_INFO_LOG_BUFFER;
		values[1] = (void*)info_log;
		options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		values[2] = (void*)JIT_BUF_SIZE;
		options[3] = CU_JIT_ERROR_LOG_BUFFER;
		values[3] = (void*)error_log;
		options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		values[4] = (void*)JIT_BUF_SIZE;
		options[5] = CU_JIT_LOG_VERBOSE;
		values[5] = (void*)1;
		cuda_error_drv( cuLinkCreate(6, options, values, &linkState) );
	}

};

class RemoteAssistant {
	friend void* _TaskRecvThread( void* );
public:
	RemoteAssistant(int *argc, char ***argv);
	virtual ~RemoteAssistant();
	void Finalize();
	void Init();
	void Run();
	void CudaDeviceSetCacheConfig();
	void CudaGetDeviceProperties();
	void CudaGetErrorString();
	void CudaStreamCreate();
	void CudaStreamDestroy();
	void CudaStreamSynchronize();
	void CudaMalloc();
	void CudaMemset();
	void CudaFree();
	void CudaMemcpyAsync();
	size_t CudaLinkAddData(char **ptxCode, size_t *size);
	void CudaConfigureCall();
	void CudaSetupArgument();
	void CudaModuleLoad();
	void CudaModuleGetFunction();
	void CudaModuleGetSymbol();
	void CudaLaunch();
	void CudaGetLastError();
	void CudaDeviceSynchronize();
	void CudaDeviceReset();
	void CudaDeviceSetLimit();
	void CudaEventCreate();
	void CudaEventDestroy();
	void CudaEventRecord();
	void CudaEventSynchronize();
	void CudaEventElapsedTime();
	static bool UseRDMA() { return m_UseRDMA; };
private:
	char *m_Buffer;
	MPI_Status m_Status;
	std::vector<CUmodule> m_CUmodule;
	int m_LocalNodeId;
	int m_DeviceId;

	std::map<cudaStream_t, Stream*> m_Streams;
	int m_StreamId;

	//kernel launch parameters
	std::map<int,char *> m_KernelParamBuffer;
	std::map<int,size_t> m_KernelParamBufferSize;
	std::map<int,size_t> m_MaxKernelParamBufferSize;
	std::map<int,dim3> m_KernelGridDim;
	std::map<int,dim3> m_KernelBlockDim;
	std::map<int,size_t> m_KernelSharedMem;
	std::map<int,CUstream> m_KernelStream;

	//MPI Communicator
	MPI_Comm m_HostComm;

	//cudaCallback ID
	int m_CudaCallbackId;

	//RDMA flag
	static bool m_UseRDMA;

	//task queue
	char* m_TaskQue[TASK_QUE_SIZE];
	MPI_Status *m_StatusQue;
	int m_TaskQueFront, m_TaskQueRear;
	sem_t m_FreeTaskSlotCount, m_ReadyTaskSlotCount;
	pthread_t m_TaskRecvThread;
};
extern int gloableDeviceId;
#endif /* REMOTEASSISTANT_H_ */
