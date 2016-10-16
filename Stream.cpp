/*
 * Stream.cpp
 *
 *  Created on: 2015-9-9
 *      Author: makai
 */

#include "Stream.h"
#include "Functions.h"
#include "Types.h"
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include "RemoteAssistant.h"
extern int globalDeviceId;
void* _StreamTaskSubmitThread(void* arg) {
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	cudaSetDevice(0);
	Stream* streamObj = (Stream*)arg;
	StreamTask* taskp;
	StreamTaskKernel* kernelTaskp;
	StreamTaskMemcpy* memcpyTaskp;
	cudaMemcpyKind kind;
	streamObj->m_HostRecvBufQueFront = 0;
	streamObj->m_HostSendBufQueRear = 0;
	streamObj->m_TaskQueFront = 0;
	bool streamFlag = true;
	while(streamFlag) {
		//pthread_testcancel();
		taskp = streamObj->TaskPop();
		//pthread_testcancel();
		switch(taskp->GetTaskType()) {
		case StreamTaskType_Kernel:
			kernelTaskp = static_cast<StreamTaskKernel*>(taskp);	
			cuda_error_drv( kernelTaskp->LaunchKernel() );
			delete kernelTaskp;
			break;
		case StreamTaskType_Memcpy:
			//printf("StreamTask memcpy\n");
			memcpyTaskp = static_cast<StreamTaskMemcpy*>(taskp);
			kind = memcpyTaskp->GetKind();
			if(kind == cudaMemcpyHostToDevice)
				streamObj->MemcpyHostToDevice(memcpyTaskp);
			else if(kind == cudaMemcpyDeviceToHost)
				streamObj->MemcpyDeviceToHost(memcpyTaskp);
			delete memcpyTaskp;
			break;
		case StreamTaskType_Synchronize:
		{	//printf("StreamTask streamsync start on\n");
			cudaStream_t stream;
			if(streamObj->m_StreamFlag == NullStreamFlag)
				stream = NULL;
			else
				stream = streamObj->m_Stream;
			//printf("StreamTask streamsync before cudaStreamSynchronize on %d\n",globalDeviceId);
			streamObj->m_LastStreamError = cudaStreamSynchronize(stream);
			//printf("StreamTask streamsync after cudaStreamSynchronize on %d\n",globalDeviceId);
			sem_post(&streamObj->m_CompleteSyncCount);
			delete taskp;
			break;
		}
		case StreamTaskType_Destroy:
			cudaStreamDestroy(streamObj->m_Stream);
			streamFlag = false;
			break;
		case StreamTaskType_EventRecord:
		{
			cudaStream_t stream;
			if(streamObj->m_StreamFlag == NullStreamFlag)
				stream = NULL;
			else
				stream = streamObj->m_Stream;
			cudaEvent_t event = streamObj->m_event;
			streamObj->m_LastStreamError = cudaEventRecord(event,stream);
			sem_post(&streamObj->m_CompleteRecordCount);
			delete taskp;
			break;
		}	
		}
	}
	return NULL;
}

void CUDART_CB _CallbackInit(cudaStream_t stream, cudaError_t status, void* userData) {
	Stream* _stream = (Stream*)userData;
	_stream->m_HostSendBufQueFront = 0;
	_stream->m_HostRecvBufQueRear = 0;
}

Stream::Stream(cudaStream_t stream, int srcProc, MPI_Comm comm, int tag, enum GC_StreamFlag flag,int _rank) :
		m_Stream(stream), m_SrcProc(srcProc),
		m_Comm(comm), m_MsgTag(tag), m_StreamFlag(flag),rank(_rank),
		m_HostRecvBufQueFront(0), m_HostRecvBufQueRear(0),
		m_HostSendBufQueFront(0), m_HostSendBufQueRear(0),
		m_LastStreamError(cudaSuccess),
		m_StreamDestroyStatus(false),
		m_TaskQueFront(0), m_TaskQueRear(0)
{
	// TODO Auto-generated constructor stub
	sem_init(&m_CompleteSyncCount, 0, 0);
	sem_init(&m_CompleteRecordCount, 0, 0);
#ifndef _NO_PIPELINE
	for(int i = 0; i < HOST_BUFFER_COUNT; i++) {
		cuda_error( cudaMallocHost((void **)(m_HostRecvBufQue + i), HOST_BUFFER_INIT_SIZE) );
		cuda_error( cudaMallocHost((void **)(m_HostSendBufQue + i), HOST_BUFFER_INIT_SIZE) );
		m_HostRecvBufSize[i] = HOST_BUFFER_INIT_SIZE;
		m_HostSendBufSize[i] = HOST_BUFFER_INIT_SIZE;
		m_HostSendCount[i] = 0;
		m_SendReq[i] = MPI_REQUEST_NULL;
		m_RecvReq[i] = MPI_REQUEST_NULL;
		//printf("Stream Recv buf = %x\n",m_HostRecvBufQue + i);
		//printf("Stream Send buf = %x\n",m_HostSendBufQue + i);
		//cuda_error( cudaEventCreateWithFlags(m_RecvEvents + i, cudaEventBlockingSync | cudaEventDisableTiming) );
	}
#endif
	sem_init(&m_RecvReadyCount, 0, HOST_BUFFER_COUNT);
	cuda_error( cudaStreamAddCallback(m_Stream, _CallbackInit, this, 0) );
	m_StreamTaskQue = new StreamTask*[STREAM_TASK_QUE_SIZE];
	sem_init(&m_TaskCount, 0, 0);
	sem_init(&m_TaskFreeCount, 0, STREAM_TASK_QUE_SIZE);
	sem_init(&m_SendCallbackSem,0,1);
	sem_init(&m_RecvCallbackSem,0,1);
	syscall_error( sem_init(&m_SendableCount, 0, HOST_BUFFER_COUNT) );

	syscall_error( pthread_create(&m_StreamTaskSubmitThread, NULL, _StreamTaskSubmitThread, this) );
	//pthread_detach(m_StreamTaskSubmitThread);
}

Stream::~Stream() {
	// TODO Auto-generated destructor stub
	//printf("~Stream:%d at 1\n",m_MsgTag);
	if(!m_StreamDestroyStatus)
		pthread_cancel(m_StreamTaskSubmitThread);
	pthread_join(m_StreamTaskSubmitThread, NULL);
	//printf("~Stream:%d at 2\n",m_MsgTag);
	for(int i = 0; i < HOST_BUFFER_COUNT; i++) {
		//if(*(m_RecvReq + i) != MPI_REQUEST_NULL)
		//	MPI_Cancel(m_RecvReq + i);
		if(m_RecvReq[i] != MPI_REQUEST_NULL)
			MPI_Request_free(m_RecvReq + i);
		MPI_Wait(m_SendReq + i, MPI_STATUS_IGNORE);
	}
	//printf("~Stream:%d at 3\n",m_MsgTag);
	/* 
	sem_destroy(&m_CompleteSyncCount);
	sem_destroy(&m_SendableCount);
	sem_destroy(&m_RecvReadyCount);
	for(int i = 0; i < HOST_BUFFER_COUNT; i++) {
		cudaFreeHost(m_HostRecvBufQue[i]);
		cudaFreeHost(m_HostSendBufQue[i]);
		//cudaEventDestroy(m_RecvEvents[i]);
	}
	sem_destroy(&m_TaskCount);
	sem_destroy(&m_TaskFreeCount);
	sem_destroy(&m_RecvCallbackSem);
	sem_destroy(&m_SendCallbackSem);
	printf("~Stream:%d at 4\n",m_MsgTag);
	*/
}


void Stream::MemcpyAsync(void* dptr, size_t count, cudaMemcpyKind kind,int _threadId) {
	StreamTaskMemcpy* memcpyTask = new StreamTaskMemcpy(StreamTaskType_Memcpy, dptr, count, kind, m_Stream, _threadId, m_MsgTag);
	TaskPush((StreamTask*)memcpyTask);
}

void Stream::LaunchKernel(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, size_t argBufSize, void* argBuf, CUfunction func) {
	if(m_StreamFlag == NullStreamFlag)
		stream = NULL;
	StreamTaskKernel* kernelTask = new StreamTaskKernel(StreamTaskType_Kernel, gridDim, blockDim, sharedMem, stream, argBufSize, argBuf, func);
	TaskPush((StreamTask*)kernelTask);
}

cudaError_t Stream::Synchronize() {
	//printf("Stream Synchronize begin on %d\n",globalDeviceId);
	StreamTask* syncTask = new StreamTask(StreamTaskType_Synchronize);
	TaskPush((StreamTask*)syncTask);
	//printf("Stream Synchronize push on %d\n",globalDeviceId);
	sem_wait(&m_CompleteSyncCount);
	//printf("Stream Synchronize end on %d\n",globalDeviceId);
	return m_LastStreamError;
}

cudaError_t Stream::EventRecord(cudaEvent_t event) {
	StreamTask* eventRecordTask = new StreamTask(StreamTaskType_EventRecord);
	m_event = event;
	TaskPush((StreamTask*)eventRecordTask);
	sem_wait(&m_CompleteRecordCount);
	return m_LastStreamError;
}


void Stream::Destroy() {
	StreamTask* destroyTask = new StreamTask(StreamTaskType_Destroy);
	TaskPush((StreamTask*)destroyTask);
	m_StreamDestroyStatus = true;
}

void CUDART_CB _RecvablePostCallback(cudaStream_t stream, cudaError_t status, void *data) {
	Stream* _stream = (Stream*)data;
	sem_wait(&_stream->m_RecvCallbackSem);
	//printf("MPI_Start at |%d|,rank = %d,index = %d\n",__LINE__,_stream->rank,_stream->m_HostRecvBufQueRear);
	//mpi_error( MPI_Start(_stream->m_RecvReq + _stream->m_HostRecvBufQueRear) );
	_stream->m_HostRecvBufQueRear = (_stream->m_HostRecvBufQueRear + 1) & (HOST_BUFFER_COUNT - 1);
	sem_post(&_stream->m_RecvReadyCount);
	sem_post(&_stream->m_RecvCallbackSem);
}

void CUDART_CB _RecvablePostCallbackClean(cudaStream_t stream, cudaError_t status, void *data) {
	Stream* _stream = (Stream*)data;
	sem_post(&_stream->m_RecvReadyCount);
}

void Stream::MemcpyHostToDevice( StreamTaskMemcpy* memcpyTask ) {
	cudaStream_t stream;
	if(m_StreamFlag == NullStreamFlag)
		stream = NULL;
	else
		stream = m_Stream;
	void* buf = NULL;
	void* dptr = memcpyTask->GetPtr();
	m_threadId = memcpyTask->GetThreadId();
	buf = (void *)malloc(memcpyTask->GetCount());
	//printf("Stream before recv buf=%lx,count=%lx,srcProc=%d,tag=%lx\n",buf,memcpyTask->GetCount(),m_SrcProc,m_threadId<<16|m_MsgTag);
	mpi_error( MPI_Recv(buf, memcpyTask->GetCount(), MPI_BYTE, m_SrcProc,m_threadId << 16 | m_MsgTag , m_Comm, MPI_STATUS_IGNORE) );
	//printf("Stream after recv\n");
	timeval start_time,end_time;
	gettimeofday(&start_time,0);
	cudaMemcpyAsync(dptr, buf, memcpyTask->GetCount(), cudaMemcpyHostToDevice, stream); 
	cuda_error( cudaStreamSynchronize(stream) );
	gettimeofday(&end_time,0);
	double timeUse=(double)1000000*(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
	double bandwith = (double)memcpyTask->GetCount() * 1000000 / (1 << 20) / timeUse;

	printf("memcpyHostToDevice size=%d bandWith=%f time=%f\n",memcpyTask->GetCount(),bandwith,timeUse);
	//printf("Stream sync\n");
	free(buf);
	//printf("Stream memcpy host to device on %d end\n",globalDeviceId);
}

void CUDART_CB _SendablePostCallback(cudaStream_t stream, cudaError_t status, void *data) {
	cuda_error( status );
	Stream* _stream = (Stream*)data;
	sem_wait(&_stream->m_SendCallbackSem);
	_stream->SendData();
	sem_post(&_stream->m_SendCallbackSem);
}

void Stream::MemcpyDeviceToHost( StreamTaskMemcpy* memcpyTask ) {
	cudaStream_t stream;
	if(m_StreamFlag == NullStreamFlag)
		stream = NULL;
	else
		stream = m_Stream;
	void* buf;
	m_threadId = memcpyTask->GetThreadId();
	buf = (void *)malloc(memcpyTask->GetCount()) ;
	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,stream);
	cudaMemcpyAsync(buf,memcpyTask->GetPtr(), memcpyTask->GetCount(),cudaMemcpyDeviceToHost,stream);
	cudaStreamSynchronize(stream);
	cudaEventRecord(end,stream);
	cudaEventSynchronize(end);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms,start,end);
	printf("memcpyDeviceToHost size=%d time=%.3f\n",memcpyTask->GetCount(),ms);

	MPI_Send(buf,memcpyTask->GetCount(),MPI_BYTE,m_SrcProc,m_threadId << 16 | m_MsgTag,m_Comm);
	free(buf);
}
