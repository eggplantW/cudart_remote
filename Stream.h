#ifndef STREAM_H_
#define STREAM_H_

#include <vector>
#include <queue>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <string.h>

#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Functions.h"
#include "Defines.h"
#include "Types.h"

#define HOST_BUFFER_COUNT 2

#define STREAM_TASK_QUE_SIZE 4096

enum StreamTaskType { StreamTaskType_Kernel, StreamTaskType_Memcpy, StreamTaskType_Synchronize, StreamTaskType_Destroy};

class StreamTask {
public:
	StreamTask(StreamTaskType taskType): m_TaskType(taskType) { }
	virtual ~StreamTask() { }
	StreamTaskType GetTaskType() { return m_TaskType; }
private:
	 StreamTaskType m_TaskType;
};

class StreamTaskKernel : public StreamTask {
public:
	StreamTaskKernel(StreamTaskType taskType, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, size_t argBufSize, void* argBuf, CUfunction func) :
		StreamTask(taskType), m_GridDim(gridDim), m_BlockDim(blockDim), m_SharedMem(sharedMem), m_Stream(stream), m_ArgBufSize(argBufSize), m_Func(func) {
		m_ArgBuf = malloc( argBufSize );
		memcpy(m_ArgBuf, argBuf, argBufSize);
	}
	virtual ~StreamTaskKernel() { free(m_ArgBuf); }

	CUresult LaunchKernel() {
		void *kernel_launch_config[] = {
				CU_LAUNCH_PARAM_BUFFER_POINTER, m_ArgBuf,
				CU_LAUNCH_PARAM_BUFFER_SIZE, &m_ArgBufSize,
				CU_LAUNCH_PARAM_END
		};
		CUresult result;
		result = cuLaunchKernel(m_Func, m_GridDim.x, m_GridDim.y, m_GridDim.z, m_BlockDim.x, m_BlockDim.y, m_BlockDim.z,
				m_SharedMem, m_Stream, NULL, kernel_launch_config);
		return result;
	}
private:
	dim3 m_GridDim, m_BlockDim;
	size_t m_SharedMem;
	cudaStream_t m_Stream;
	size_t m_ArgBufSize;
	void* m_ArgBuf;
	CUfunction m_Func;
};

class StreamTaskMemcpy : public StreamTask {
public:
	StreamTaskMemcpy(StreamTaskType taskType, void* dptr, size_t size, cudaMemcpyKind kind, cudaStream_t stream,int threadId,int streamId):
		StreamTask(taskType), m_Dptr(dptr), m_Size(size), m_Kind(kind), m_Stream(stream),m_threadId(threadId),m_streamId(streamId) { }
	virtual ~StreamTaskMemcpy() { }
	cudaError_t Memcpy(void* hptr) {
		cudaError_t resu;
		if(m_Kind == cudaMemcpyHostToDevice)
			resu = cudaMemcpyAsync(m_Dptr, hptr, m_Size, m_Kind, m_Stream);
		else if(m_Kind == cudaMemcpyDeviceToHost)
			resu = cudaMemcpyAsync(hptr, m_Dptr, m_Size, m_Kind, m_Stream);
		else
			resu = cudaErrorInvalidMemcpyDirection;
		return resu;
	}
	size_t GetCount() { return m_Size; };
	cudaMemcpyKind GetKind() { return m_Kind; }
	void* GetPtr() { return m_Dptr; }
	cudaStream_t GetStream() { return m_Stream; }
	int GetThreadId()	{	return m_threadId;}
private:
	void* m_Dptr;
	size_t m_Size;
	cudaMemcpyKind m_Kind;
	cudaStream_t m_Stream;
	int m_threadId;
	int m_streamId;
};

class Stream {
	friend void CUDART_CB _RecvablePostCallbackClean(cudaStream_t stream, cudaError_t status, void *data);
	friend void CUDART_CB _RecvablePostCallback(cudaStream_t stream, cudaError_t status, void *data);
	friend void CUDART_CB _SendablePostCallback(cudaStream_t stream, cudaError_t status, void *data);
	friend void CUDART_CB _CallbackInit(cudaStream_t stream, cudaError_t status, void* userData);
	friend void* _StreamTaskSubmitThread( void* );
	friend void* _RecvThread(void* arg);
public:
	Stream(cudaStream_t stream, int srcProc, MPI_Comm comm, int tag, enum GC_StreamFlag flag,int rank);
	virtual ~Stream();
	void MemcpyAsync(void* dptr, size_t count, cudaMemcpyKind kind, int threadId);
	void LaunchKernel(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, size_t argBufSize, void* argBuf, CUfunction func);
	cudaError_t Synchronize();
	void Destroy();
	int GetSrcProc() { return m_SrcProc; }
	enum GC_StreamFlag GetStreamFlag() { return m_StreamFlag; }

private:
	cudaStream_t m_Stream;
	int m_SrcProc;//source process id
	MPI_Comm m_Comm;//parrent comm
	int m_MsgTag;//stream id
	enum GC_StreamFlag m_StreamFlag;//NULL Default NonBlocking 

	cudaError_t m_LastStreamError;
	sem_t m_CompleteSyncCount;


	bool m_StreamDestroyStatus;

	//buffers for receive
	char* m_HostRecvBufQue[HOST_BUFFER_COUNT];
	size_t m_HostRecvBufSize[HOST_BUFFER_COUNT];
	int m_HostRecvBufQueFront, m_HostRecvBufQueRear;
	cudaEvent_t m_RecvEvents[HOST_BUFFER_COUNT];
	MPI_Request m_RecvReq[HOST_BUFFER_COUNT];
	sem_t m_RecvReadyCount;
	sem_t m_RecvCallbackSem;

	//buffers for send
	char* m_HostSendBufQue[HOST_BUFFER_COUNT];
	size_t m_HostSendBufSize[HOST_BUFFER_COUNT];
	size_t m_HostSendCount[HOST_BUFFER_COUNT];
	int m_HostSendBufQueFront, m_HostSendBufQueRear;
	MPI_Request m_SendReq[HOST_BUFFER_COUNT];
	sem_t m_SendableCount;
	int m_HostSendTag[HOST_BUFFER_COUNT];
	sem_t m_SendCallbackSem;

	pthread_t m_StreamTaskSubmitThread;

	//StreamTaskQue
	//std::queue<StreamTask*>* m_StreamTaskQue;
	//pthread_mutex_t m_StreamTaskQueLck;
	//get the top task in the task queue. This function will stuck until the queue is not empty.
	//StreamTask* StreamTaskQuePop();
	StreamTask** volatile m_StreamTaskQue;
	int m_TaskQueFront, m_TaskQueRear;
	sem_t m_TaskCount, m_TaskFreeCount;

	int m_threadId;
	int rank;

	void MemcpyHostToDevice( StreamTaskMemcpy* memcpyTask );

	void MemcpyDeviceToHost( StreamTaskMemcpy* memcpyTask );
	void SendData() {
		//printf("mpi_isend at %s:%d front = %d,tag = %d\n",__FILE__,__LINE__,m_HostSendBufQueFront,m_HostSendTag[m_HostSendBufQueFront]);
		mpi_error( MPI_Send(m_HostSendBufQue[m_HostSendBufQueFront], m_HostSendCount[m_HostSendBufQueFront], MPI_BYTE, m_SrcProc, m_HostSendTag[m_HostSendBufQueFront], m_Comm) );
		sem_post(&m_SendableCount);
		m_HostSendBufQueFront = (m_HostSendBufQueFront + 1) & (HOST_BUFFER_COUNT - 1);
	}
	void TaskPush(StreamTask* task) {
		sem_wait(&m_TaskFreeCount);
		m_StreamTaskQue[m_TaskQueRear] = task;
		sem_post(&m_TaskCount);
		m_TaskQueRear = (m_TaskQueRear + 1) & (STREAM_TASK_QUE_SIZE - 1);
	}
	StreamTask* TaskPop() {
		sem_wait(&m_TaskCount);
		pthread_testcancel();
		StreamTask* task = m_StreamTaskQue[m_TaskQueFront];
		sem_post(&m_TaskFreeCount);
		m_TaskQueFront = (m_TaskQueFront + 1) & (STREAM_TASK_QUE_SIZE - 1);
		return task;
	}
};

#endif /* STREAM_H_ */
