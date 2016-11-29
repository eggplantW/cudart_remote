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
		//	printf("StreamTask streamsync before cudaStreamSynchronize on %d\n",globalDeviceId);
			streamObj->m_LastStreamError = cudaStreamSynchronize(stream);
		//	printf("StreamTask streamsync after cudaStreamSynchronize on %d\n",globalDeviceId);
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
	for(int i = 0; i < HOST_BUFFER_COUNT; i++) 
	{
		m_HostRecvBufQue[i] = (char *)malloc(sizeof(char)*HOST_BUFFER_INIT_SIZE);
		m_HostSendBufQue[i] = (char *)malloc(sizeof(char)*HOST_BUFFER_INIT_SIZE);
		m_HostRecvBufSize[i] = HOST_BUFFER_INIT_SIZE;
		m_HostSendBufSize[i] = HOST_BUFFER_INIT_SIZE;
		m_HostSendCount[i] = 0;
		m_SendReq[i] = MPI_REQUEST_NULL;
		m_RecvReq[i] = MPI_REQUEST_NULL;
		//printf("Stream Recv buf = %x\n",m_HostRecvBufQue + i);
		//printf("Stream Send buf = %x\n",m_HostSendBufQue + i);
		//cuda_error( cudaEventCreateWithFlags(m_RecvEvents + i, cudaEventBlockingSync | cudaEventDisableTiming) );
	}
	sem_init(&m_RecvReadyCount, 0, HOST_BUFFER_COUNT);
	sem_init(&m_SendableCount, 0, HOST_BUFFER_COUNT);
	cuda_error( cudaStreamAddCallback(m_Stream, _CallbackInit, this, 0) );
	m_StreamTaskQue = new StreamTask*[STREAM_TASK_QUE_SIZE];
	sem_init(&m_TaskCount, 0, 0);
	sem_init(&m_TaskFreeCount, 0, STREAM_TASK_QUE_SIZE);
	sem_init(&m_SendCallbackSem,0,1);
	sem_init(&m_RecvCallbackSem,0,1);
	

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
	
	sem_destroy(&m_CompleteSyncCount);
	sem_destroy(&m_SendableCount);
	sem_destroy(&m_RecvReadyCount);
	for(int i = 0; i < HOST_BUFFER_COUNT; i++) {
		free(m_HostRecvBufQue[i]);
		free(m_HostSendBufQue[i]);
		//cudaEventDestroy(m_RecvEvents[i]);
	}
	sem_destroy(&m_TaskCount);
	sem_destroy(&m_TaskFreeCount);
	sem_destroy(&m_RecvCallbackSem);
	sem_destroy(&m_SendCallbackSem);
	//printf("~Stream:%d at 4\n",m_MsgTag);
	
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
//	printf("MPI_Start at |%d|,threadId = %d,index = %d\n",__LINE__,_stream->m_threadId,_stream->m_HostRecvBufQueRear);
	mpi_error( MPI_Start(_stream->m_RecvReq + _stream->m_HostRecvBufQueRear) );
	_stream->m_HostRecvBufQueRear = (_stream->m_HostRecvBufQueRear + 1) & (HOST_BUFFER_COUNT - 1);
	sem_post(&_stream->m_RecvReadyCount);
	sem_post(&_stream->m_RecvCallbackSem);
}

void CUDART_CB _RecvablePostCallbackClean(cudaStream_t stream, cudaError_t status, void *data) {
	Stream* _stream = (Stream*)data;
//	printf("MPI_Start at |%d|,threadId = %d\n",__LINE__,_stream->m_threadId);
	sem_post(&_stream->m_RecvReadyCount);
}

void Stream::MemcpyHostToDevice( StreamTaskMemcpy* memcpyTask ) {
	cudaStream_t stream;
	if(m_StreamFlag == NullStreamFlag)
		stream = NULL;
	else
		stream = m_Stream;
	size_t count = memcpyTask->GetCount();
	char* dptr = (char*)memcpyTask->GetPtr();
	m_threadId = memcpyTask->GetThreadId();
	void* buf = NULL;

	for(int i = 0;i < HOST_BUFFER_COUNT;i ++)
		sem_wait(&m_RecvReadyCount);
	for(int i = 0;i < HOST_BUFFER_COUNT;i ++)
		sem_post(&m_RecvReadyCount);
	m_HostRecvBufQueFront = 0;
	m_HostRecvBufQueRear = 0;

	int bufCount;
	if(count % HOST_BUFFER_INIT_SIZE == 0)
		bufCount = count / HOST_BUFFER_INIT_SIZE;
	else
		bufCount = count / HOST_BUFFER_INIT_SIZE + 1;
	if (bufCount >= HOST_BUFFER_COUNT)
		bufCount = HOST_BUFFER_COUNT;
	for(int i = 0;i < bufCount;i ++)
	{
		if(*(m_RecvReq + i) != MPI_REQUEST_NULL)
			mpi_error(MPI_Request_free(m_RecvReq + i));
		mpi_error(MPI_Recv_init(m_HostRecvBufQue[i],HOST_BUFFER_INIT_SIZE,MPI_BYTE,m_SrcProc,m_threadId<<16|m_MsgTag,m_Comm,m_RecvReq+i));
		mpi_error(MPI_Start(m_RecvReq + i));
	}

	while(count > HOST_BUFFER_INIT_SIZE * HOST_BUFFER_COUNT)
	{
		buf = m_HostRecvBufQue[m_HostRecvBufQueFront];
		sem_wait(&m_RecvReadyCount);
		mpi_error(MPI_Wait(m_RecvReq + m_HostRecvBufQueFront,MPI_STATUS_IGNORE));
		cuda_error(cudaMemcpyAsync(dptr,buf,HOST_BUFFER_INIT_SIZE,cudaMemcpyHostToDevice,stream));
		cuda_error(cudaStreamAddCallback(stream,_RecvablePostCallback,this,0));
		m_HostRecvBufQueFront = (m_HostRecvBufQueFront + 1)&(HOST_BUFFER_COUNT - 1);
		count = count - HOST_BUFFER_INIT_SIZE;
		dptr = dptr + HOST_BUFFER_INIT_SIZE;
	}
	while(bufCount > 0)
	{
		bufCount --;
		size_t _count;
		if(count > HOST_BUFFER_INIT_SIZE)
			_count = HOST_BUFFER_INIT_SIZE;
		else
			_count = count;
		buf = m_HostRecvBufQue[m_HostRecvBufQueFront];
		sem_wait(&m_RecvReadyCount);
		mpi_error(MPI_Wait(m_RecvReq + m_HostRecvBufQueFront,MPI_STATUS_IGNORE));
		cuda_error(cudaMemcpyAsync(dptr,buf,_count,cudaMemcpyHostToDevice,stream));
		cuda_error(cudaStreamAddCallback(stream,_RecvablePostCallbackClean,this,0));
		m_HostRecvBufQueFront = (m_HostRecvBufQueFront + 1)&(HOST_BUFFER_COUNT - 1);
		count = count - _count;
		dptr = dptr + _count;
	}
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
	size_t count = memcpyTask->GetCount();
	char* dptr = (char*)memcpyTask->GetPtr();
	m_threadId = memcpyTask->GetThreadId();
	void* buf = NULL;
	while(count > 0)
	{
		sem_wait(&m_SendableCount);
		size_t _count;
		if(count > HOST_BUFFER_INIT_SIZE)
			_count = HOST_BUFFER_INIT_SIZE;
		else
			_count = count;
		buf = m_HostSendBufQue[m_HostSendBufQueRear];
		mpi_error(MPI_Wait(m_SendReq + m_HostSendBufQueRear,MPI_STATUS_IGNORE));
		m_HostSendCount[m_HostSendBufQueRear] = _count;
		m_HostSendTag[m_HostSendBufQueRear] = m_threadId << 16 | m_MsgTag;
		cuda_error(cudaMemcpyAsync(buf,dptr,_count,cudaMemcpyDeviceToHost,stream));
		cuda_error(cudaStreamAddCallback(stream,_SendablePostCallback,this,0));
		m_HostSendBufQueRear = (m_HostSendBufQueRear + 1) % HOST_BUFFER_COUNT;
		count = count - _count;
		dptr = dptr + _count;
	}
}
