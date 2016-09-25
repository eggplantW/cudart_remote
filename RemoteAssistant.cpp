/*
 * RemoteAssistant.cpp
 *
 *  Created on: 2015-6-16
 *      Author: makai
 */

#include "RemoteAssistant.h"
#include "Types.h"
#include "Defines.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>//for signal caught
using namespace std;
int globalDeviceId;
void* _TaskRecvThread( void* arg ) {
	RemoteAssistant* rast = (RemoteAssistant*)arg;
	char** taskQue = rast->m_TaskQue;
	MPI_Status *statQue = rast->m_StatusQue;
	int taskQueRear = 0;
	sem_t* freeTaskSlotCount = &(rast->m_FreeTaskSlotCount);
	sem_t* readyTaskSlotCount = &(rast->m_ReadyTaskSlotCount);
	MPI_Comm comm = rast->m_HostComm;
	while(true) {
		sem_wait(freeTaskSlotCount);
		mpi_error( MPI_Recv(taskQue[taskQueRear], TASK_BUF_SIZE, MPI_BYTE, MPI_ANY_SOURCE, NullTag, comm, statQue + taskQueRear) );
		taskQueRear = (taskQueRear + 1) % TASK_QUE_SIZE;
		sem_post(readyTaskSlotCount);
	}
	return NULL;
}

bool RemoteAssistant::m_UseRDMA = false;

RemoteAssistant::RemoteAssistant(int *argc, char ***argv): m_Buffer(NULL), m_CudaCallbackId(0), m_StreamId(1){
	// TODO Auto-generated constructor stub
	int threadStatus;
	mpi_error( MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &threadStatus) );
	if(threadStatus != MPI_THREAD_MULTIPLE)
		cout << "thread error" << endl;
	mpi_error( MPI_Comm_rank(MPI_COMM_WORLD, &m_LocalNodeId) );
	mpi_error( MPI_Comm_get_parent(&m_HostComm) );
	globalDeviceId = m_LocalNodeId;
	Init();
	//printf("RemoteAssistant parentComm = %x\n",m_HostComm);
	for(int i = 0; i < TASK_QUE_SIZE; i++)
		m_TaskQue[i] = new char[TASK_BUF_SIZE];
	m_StatusQue = (MPI_Status*)malloc(TASK_QUE_SIZE * sizeof(MPI_Status));
	m_TaskQueFront = 0;
	m_TaskQueRear = 0;
	sem_init(&m_ReadyTaskSlotCount, 0, 0);
	sem_init(&m_FreeTaskSlotCount, 0, TASK_QUE_SIZE);
}

RemoteAssistant::~RemoteAssistant() {
	//printf("0 %d\n",m_LocalNodeId);
	//printf("~RemoteAssistant:0.1 %d\n",m_LocalNodeId);
	pthread_cancel(m_TaskRecvThread);
	//printf("~RemoteAssistant:0.2 %d\n",m_LocalNodeId);
	//for(int i = 0; i < TASK_QUE_SIZE; i++)
	//	delete[] m_TaskQue[i];
	//printf("~RemoteAssistant:0.3 %d\n",m_LocalNodeId);
	//free(m_StatusQue);
	//if(m_KernelParamBuffer)
	//	delete [] m_KernelParamBuffer;
	//printf("~RemoteAssistant:1 %d\n",m_LocalNodeId);
	for(map<cudaStream_t, Stream*>::iterator it = m_Streams.begin(); it != m_Streams.end(); it++)
		delete it->second;
	//printf("~RemoteAssistant:2 %d\n",m_LocalNodeId);
	MPI_Barrier(MPI_COMM_WORLD);
	//printf("~RemoteAssistant:3 %d\n",m_LocalNodeId);
	//if(m_LocalNodeId == 0)
	//	mpi_error( MPI_Send(NULL, 0, MPI_BYTE, 0, 0, m_HostComm) );
	//sleep(100);
	MPI_Finalize();
	//printf("~RemoteAssistant:4 %d\n",m_LocalNodeId);
}

void RemoteAssistant::Finalize() {
	// TODO Auto-generated destructor stub

}

void RemoteAssistant::CudaDeviceSetCacheConfig() {
	CudaDeviceSetCacheConfigMsg_t* msg = (CudaDeviceSetCacheConfigMsg_t*)m_Buffer;
	cudaError_t cudaErr = cudaDeviceSetCacheConfig(msg->cacheConfig);
	mpi_error( MPI_Send(&cudaErr, sizeof(cudaErr), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}


void RemoteAssistant::CudaGetDeviceProperties() {
	CudaGetDevicePropertiesMsg_t* msg = (CudaGetDevicePropertiesMsg_t*)m_Buffer;
	CudaGetDevicePropertiesAckMsg_t amsg;
	amsg.cudaStat = cudaGetDeviceProperties(&amsg.prop, 0);
	mpi_error( MPI_Send(&amsg, sizeof(amsg), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaGetErrorString() {
	CudaGetErrorStringMsg_t* msg = (CudaGetErrorStringMsg_t*)m_Buffer;
	const char* errString = cudaGetErrorString(msg->err);
	int stringLen = strlen(errString);
	mpi_error( MPI_Send(errString, stringLen+1, MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaStreamCreate() {
	CudaStreamCreateMsg_t *msg = (CudaStreamCreateMsg_t*)m_Buffer;
	CudaStreamCreateAckMsg_t amsg;
	cudaStream_t stream;
	//printf("RemoteAssistant streamCreate at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	if(msg->flag == NullStreamFlag || msg->flag == NonblockingStreamFlag )
		amsg.status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	else
		amsg.status = cudaStreamCreate(&stream);
	if(amsg.status == cudaSuccess) {
			m_Streams[stream] = new Stream(stream, m_Status.MPI_SOURCE, m_HostComm, m_StreamId, msg->flag,m_LocalNodeId);
		amsg.stream = stream;
		amsg.streamTag = m_StreamId;
		m_StreamId++;
	}
	/* 
	std::map<cudaStream_t,Stream*>::iterator It;
		char tmpStr[1024];
		char tmpS[100];
		tmpStr[0] = '\0';
		sprintf(tmpS,"Remote Stream on device %d to find stream = %x has(StreamCreate):\n",m_LocalNodeId,stream);
		strcat(tmpStr,tmpS);
		for(It = m_Streams.begin();It != m_Streams.end();It ++)
		{
			sprintf(tmpS,"streamId = %x\n",It->first);
			strcat(tmpStr,tmpS);
		}
		printf("%s\n",tmpStr);
	*/
	mpi_error( MPI_Send(&amsg, sizeof(CudaStreamCreateAckMsg_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
	//printf("RemoteAssistant streamCreate end at line:%d,threadId = %d\n",__LINE__,msg->threadId);
}

void RemoteAssistant::CudaStreamDestroy() {
	CudaStreamDestroyMsg_t* msg = (CudaStreamDestroyMsg_t*)m_Buffer;
	//note: the corresponding Stream has not been delete
	cudaError_t status;
	map<cudaStream_t, Stream*>::iterator streamIt = m_Streams.find(msg->stream);
	if(streamIt == m_Streams.end())
		status = cudaErrorInvalidResourceHandle;
	else {
		streamIt->second->Destroy();
		status = cudaSuccess;
	}
	mpi_error( MPI_Send(&status, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaStreamSynchronize() {
	CudaStreamSynchronizeMsg_t* msg = (CudaStreamSynchronizeMsg_t*)m_Buffer;
	cudaError_t status;
	//printf("RemoteAssistant on device %d before streamsync at line:%d,threadId = %d,stream = %x\n",m_LocalNodeId,__LINE__,msg->threadId,msg->stream);
	map<cudaStream_t, Stream*>::iterator streamIt = m_Streams.find(msg->stream);
	if(streamIt == m_Streams.end())
	{
		//printf("didn't find stream %x\n",msg->stream);
		status = cudaErrorInvalidResourceHandle;
	}
	else {
		//printf("RemoteAssistant on device %d streamsync at line:%d,threadId = %d,stream = %x\n",m_LocalNodeId,__LINE__,msg->threadId,msg->stream);
		status = streamIt->second->Synchronize();
	}
	/*
	std::map<cudaStream_t,Stream*>::iterator It;
		char tmpStr[1024];
		char tmpS[100];
		tmpStr[0] = '\0';
		sprintf(tmpS,"Remote Stream on device %d to find stream = %x has(StreamAsync):\n",m_LocalNodeId,msg->stream);
		strcat(tmpStr,tmpS);
		for(It = m_Streams.begin();It != m_Streams.end();It ++)
		{
			sprintf(tmpS,"streamId = %x\n",It->first);
			strcat(tmpStr,tmpS);
		}
		printf("%s\n",tmpStr);
	*/
	//printf("RemoteAssistant on device %d streamsync at line:%d,threadId = %d,stream = %x\n",m_LocalNodeId,__LINE__,msg->threadId,msg->stream);
	mpi_error( MPI_Send(&status, sizeof(status), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
	//printf("RemoteAssistant on device %d streamsync end at line:%d,threadId = %d,stream = %x\n",m_LocalNodeId,__LINE__,msg->threadId,msg->stream);
}

void RemoteAssistant::CudaMalloc() {
	CudaMallocMsg_t *msg = (CudaMallocMsg_t *)m_Buffer;
	//printf("RemoteAssistant malloc at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	void *d_ptr = NULL;
	cudaError_t cudaResu = cudaMalloc(&d_ptr, msg->size);
	CudaMallocAckMsg_t amsg = { d_ptr, cudaResu };
	mpi_error( MPI_Send(&amsg, sizeof(amsg), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaMemset(){
	CudaMemsetMsg_t *msg = (CudaMemsetMsg_t *)m_Buffer;
	cudaError_t cudaResu;
	//printf("RemoteAssistant memset dptr = %x,value = %d,size = %d\n",msg->dptr,msg->value,msg->size);
	cudaResu = cudaMemset(msg->dptr,msg->value,msg->size);
	//printf("RemoteAssistant memset befor send threadId = %d\n",msg->threadId);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
	//printf("RemoteAssistant memset after send threadId = %d\n",msg->threadId);
}

void RemoteAssistant::CudaFree() {
	CudaFreeMsg_t *msg = (CudaFreeMsg_t *)m_Buffer;
	cudaError_t cudaResu;
	//printf("RemoteAssistant cudaFree at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	cudaResu = cudaFree(msg->dptr);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaMemcpyAsync() {
	CudaMemcpyAsyncMsg_t *msg = (CudaMemcpyAsyncMsg_t*)m_Buffer;
/* 
	std::map<cudaStream_t,Stream*>::iterator It;
		char tmpStr[1024];
		char tmpS[100];
		tmpStr[0] = '\0';
		sprintf(tmpS,"Remote Stream on device %d to find stream = %x has(MemcpyAsync):\n",m_LocalNodeId,msg->stream);
		strcat(tmpStr,tmpS);
		for(It = m_Streams.begin();It != m_Streams.end();It ++)
		{
			sprintf(tmpS,"streamId = %x\n",It->first);
			strcat(tmpStr,tmpS);
		}
		printf("%s\n",tmpStr);
*/	
	//printf("RemoteAssistant memcpy at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	map<cudaStream_t, Stream*>::iterator streamIt = m_Streams.find(msg->stream);
	if(streamIt == m_Streams.end())
	{
		//printf("RemoteAssistant memasynv didn't find stream = %d,pid = %d\n",msg->stream,msg->threadId);
		cuda_error( cudaErrorInvalidResourceHandle );
	}
	streamIt->second->MemcpyAsync(msg->ptr, msg->count, msg->kind, msg->threadId);
}

void RemoteAssistant::CudaConfigureCall() {
	CudaConfigureCallMsg_t *msg = (CudaConfigureCallMsg_t *)m_Buffer;
	//printf("RemoteAssistant configurecall at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	int threadId = msg->threadId;
	m_KernelGridDim[threadId] = msg->gridDim;
	m_KernelBlockDim[threadId] = msg->blockDim;
	m_KernelSharedMem[threadId] = msg->sharedMem;
	m_KernelStream[threadId] = msg->stream;
	m_MaxKernelParamBufferSize[threadId] = 256;
	m_KernelParamBuffer[threadId] = new char[m_MaxKernelParamBufferSize[threadId]];
	m_KernelParamBufferSize[threadId] = 0;
}

void RemoteAssistant::CudaSetupArgument() {
	CudaSetupArgumentMsg_t *msg = (CudaSetupArgumentMsg_t *)m_Buffer;
	//printf("RemoteAssistant setupargu at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	int threadId = msg->threadId;
	if(msg->offset + msg->size > m_MaxKernelParamBufferSize[threadId]) {
		m_MaxKernelParamBufferSize[threadId] = (msg->offset + msg->size + 255)/256*256;
		char *tmpBuffer = new char[m_MaxKernelParamBufferSize[threadId]];
		memcpy(tmpBuffer, m_KernelParamBuffer[threadId], m_KernelParamBufferSize[threadId]);
		delete [] m_KernelParamBuffer[threadId];
		m_KernelParamBuffer[threadId] = tmpBuffer;
	}
	MPI_Status status;
	//printf("remote at |%d|,buf = %x,count = %d,src = %d,tag = %d,comm = %x\n",__LINE__,m_KernelParamBuffer[threadId]+msg->offset,msg->size,m_Status.MPI_SOURCE,msg->threadId << 16,m_HostComm);
	mpi_error( MPI_Recv(m_KernelParamBuffer[threadId]+msg->offset, msg->size, MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm, &status) );
	m_KernelParamBufferSize[threadId] = msg->offset+msg->size;

}

void RemoteAssistant::CudaModuleLoad() {
	CudaModuleLoadMsg_t* msg = (CudaModuleLoadMsg_t*)m_Buffer;
	//printf("RemoteAssistant moduleload at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	int fatCubinSize;
	MPI_Status stat;
	mpi_error( MPI_Probe(m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm, &stat) );
	mpi_error( MPI_Get_count(&stat, MPI_BYTE, &fatCubinSize) );
	void *fatCubin = malloc(fatCubinSize);
	//printf("remote at |%d|,buf = %x,count = %d,src = %d,tag = %d,comm = %x\n",__LINE__,fatCubin,fatCubinSize,stat.MPI_SOURCE,msg->threadId << 16,m_HostComm);
	mpi_error( MPI_Recv(fatCubin, fatCubinSize, MPI_BYTE, stat.MPI_SOURCE, msg->threadId << 16, m_HostComm, MPI_STATUS_IGNORE) );
	CUmodule cuModule;
	cuda_error_drv( cuModuleLoadFatBinary(&cuModule, fatCubin) );
	mpi_error( MPI_Send(&cuModule, sizeof(CUmodule), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
	free(fatCubin);
}

void RemoteAssistant::CudaModuleGetFunction() {
	CudaModuleGetFuncMsg_t *msg = (CudaModuleGetFuncMsg_t*)m_Buffer;
	//printf("RemoteAssistant moduleGetFunction at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	cudaError_t cudaErr;
	CUfunction kernel_func;
	int nameLen;
	MPI_Status stat;
	mpi_error( MPI_Probe(m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm, &stat) );
	mpi_error( MPI_Get_count(&stat, MPI_BYTE, &nameLen) );
	char *funcName = new char[nameLen];
	//printf("remote at |%d|,buf = %x,count = %d,src = %d,tag = %d,comm = %x\n",__LINE__,funcName,nameLen,stat.MPI_SOURCE,stat.MPI_TAG,m_HostComm);
	mpi_error( MPI_Recv(funcName, nameLen, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, m_HostComm, MPI_STATUS_IGNORE) );
	cuda_error_drv( cuModuleGetFunction(&kernel_func, msg->module, funcName) );
	mpi_error( MPI_Send(&kernel_func, sizeof(kernel_func), MPI_BYTE, m_Status.MPI_SOURCE, stat.MPI_TAG, m_HostComm) );
	delete[] funcName;
}

void RemoteAssistant::CudaModuleGetSymbol() {
	CudaModuleGetSymbolMsg_t* msg = (CudaModuleGetSymbolMsg_t*)m_Buffer;
	cudaError_t cudaErr;
	CUdeviceptr dptr;
	int nameLen;
	MPI_Status stat;
	mpi_error( MPI_Probe(m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm, &stat) );
	mpi_error( MPI_Get_count(&stat, MPI_BYTE, &nameLen) );
	char *symbolName = new char[nameLen];
	//printf("remote at |%d|,buf = %x,count = %d,src = %d,tag = %d,comm = %x\n",__LINE__,symbolName,nameLen,stat.MPI_SOURCE,stat.MPI_TAG,m_HostComm);
	mpi_error( MPI_Recv(symbolName, nameLen, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, m_HostComm, MPI_STATUS_IGNORE) );
	size_t size;
	cuda_error_drv( cuModuleGetGlobal(&dptr, &size, msg->module, symbolName) );
	mpi_error( MPI_Send(&dptr, sizeof(dptr), MPI_BYTE, m_Status.MPI_SOURCE, stat.MPI_TAG, m_HostComm) );
	delete[] symbolName;
}

void RemoteAssistant::CudaLaunch() {
	CudaLaunchMsg_t *msg = (CudaLaunchMsg_t *)m_Buffer;
	int threadId = msg->threadId;
	//printf("RemoteAssistant launch at line:%d,threadId = %d\n",__LINE__,msg->threadId);
	cudaError_t cudaErr;
	/*
	CUfunction kernel_func;
	int nameLen;
	MPI_Status stat;
	mpi_error( MPI_Probe(m_Status.MPI_SOURCE, CudaLaunchTag, m_HostComm, &stat) );
	mpi_error( MPI_Get_count(&stat, MPI_BYTE, &nameLen) );
	char *funcName = new char[nameLen];
	mpi_error( MPI_Recv(funcName, nameLen, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, m_HostComm, MPI_STATUS_IGNORE) );
	cuda_error_drv( cuModuleGetFunction(&kernel_func, msg->module, funcName) );
	delete [] funcName;
	*/

	//	printf("%s\n",tmpStr);
	std::map<cudaStream_t, Stream*>::iterator streamIt = m_Streams.find(m_KernelStream[threadId]);
	if(streamIt == m_Streams.end())
		cuda_error( cudaErrorInvalidResourceHandle );
	streamIt->second->LaunchKernel(m_KernelGridDim[threadId], m_KernelBlockDim[threadId], m_KernelSharedMem[threadId], m_KernelStream[threadId], m_KernelParamBufferSize[threadId], m_KernelParamBuffer[threadId], msg->hfunc);
	cudaErr = cudaSuccess;
	mpi_error( MPI_Send(&cudaErr, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}


void RemoteAssistant::CudaDeviceSynchronize() {
	CudaDeviceSynchronizeMsg_t* msg = (CudaDeviceSynchronizeMsg_t*)m_Buffer;
	cudaError_t cudaResu = cudaSuccess;
	//cudaResu = cudaDeviceSynchronize();
	//printf("remote device sync");
	std::map<cudaStream_t, Stream*>::iterator streamIt;
	for(streamIt = m_Streams.begin(); streamIt != m_Streams.end(); streamIt++) {
		//printf("RemoteAssistant device sync before\n");
		streamIt->second->Synchronize();
		//printf("RemoteAssistant device sync after\n");
	}
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaDeviceReset(){
	CudaDeviceResetMsg_t* msg = (CudaDeviceResetMsg_t*)m_Buffer;
	cudaError_t cudaResu = cudaSuccess;
	for(std::map<cudaStream_t, Stream*>::iterator streamIt = m_Streams.begin(); streamIt != m_Streams.end(); streamIt++) {
		streamIt->second->Destroy();
	}
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaDeviceSetLimit(){
	CudaDeviceSetLimitMsg_t* msg = (CudaDeviceSetLimitMsg_t *)m_Buffer;
	cudaError_t cudaResu = cudaDeviceSetLimit(msg->limit,msg->value);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaEventCreate(){
	CudaEventCreateMsg_t* msg = (CudaEventCreateMsg_t*)m_Buffer;
	cudaEvent_t event;
	cudaError_t cudaResu = cudaEventCreate(&event);
	CudaEventCreateAckMsg_t amsg;
	amsg.event = event;
	amsg.status = cudaResu;
	mpi_error( MPI_Send(&amsg, sizeof(amsg), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaEventDestroy(){
	CudaEventDestroyMsg_t* msg = (CudaEventDestroyMsg_t*)m_Buffer;
	cudaError_t cudaResu = cudaEventDestroy(msg->event);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaEventRecord(){
	CudaEventRecordMsg_t* msg = (CudaEventRecordMsg_t*)m_Buffer;
	cudaError_t cudaResu = cudaEventRecord(msg->event,msg->stream);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}

void RemoteAssistant::CudaEventSynchronize(){
	CudaEventSynchronizeMsg_t* msg = (CudaEventSynchronizeMsg_t*)m_Buffer;
	cudaError_t cudaResu = cudaEventSynchronize(msg->event);
	mpi_error( MPI_Send(&cudaResu, sizeof(cudaError_t), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
	
}

void RemoteAssistant::CudaEventElapsedTime(){
	CudaEventElapsedTimeMsg_t* msg = (CudaEventElapsedTimeMsg_t*)m_Buffer;
	float ms;
	cudaError_t cudaResu = cudaEventElapsedTime(&ms,msg->start,msg->end);
	CudaEventElapsedTimeAckMsg_t amsg;
	amsg.ms = ms;
	printf("RemoteAssistant cudaEventElapsedTime ms = %lf\n",amsg.ms);
	amsg.status = cudaResu;
	mpi_error( MPI_Send(&amsg, sizeof(amsg), MPI_BYTE, m_Status.MPI_SOURCE, msg->threadId << 16, m_HostComm) );
}
void RemoteAssistant::Init() {
	//get RDAM flag
	char *useRDMA = getenv("GC_USE_RDMA");
	if(useRDMA == NULL)
		m_UseRDMA = false;
	else
		m_UseRDMA = atoi(useRDMA);

	//init device info
	char *deviceList = getenv("GC_DEVICE_LIST");
	if(deviceList == NULL) {
		cerr << "cannot find GC_DEVICE_NUM_LIST" << endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	char device_str[4];
	int nodeId = -1;
	int deviceId = 0;
	char *deviceList_t = deviceList;
	while( sscanf(deviceList_t, "%s", device_str) > 0 ) {
		deviceList_t += strlen(device_str);
		while(*deviceList_t == ' ' || *deviceList_t == '\t')
			deviceList_t++;
		deviceId = atoi(device_str);
		nodeId++;
		if(nodeId == m_LocalNodeId)
			break;
	}
	if(nodeId < m_LocalNodeId) {
		cerr << "not enough devices" << endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	char visibleDevice[4];
	sprintf(visibleDevice, "%d", deviceId);
	if(setenv("CUDA_VISIBLE_DEVICES", visibleDevice, 1) != 0) {
		cerr << "set CUDA_VISIBLE_DEVICES failed" << endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	cudaSetDevice(0);
	cuda_error_drv( cuInit(0) );
	cudaFree(0);
}

void RemoteAssistant::Run() {
	int launchcount = 0;
	bool runFlag = true;
	bool stuckFlag = false;
	if(m_LocalNodeId == 0)
		stuckFlag = false;
	while(stuckFlag);

	syscall_error( pthread_create(&m_TaskRecvThread, NULL, _TaskRecvThread, this) );
	pthread_detach(m_TaskRecvThread);

	while(runFlag) {

		sem_wait(&m_ReadyTaskSlotCount);
		m_Buffer = m_TaskQue[m_TaskQueFront];
		m_Status = m_StatusQue[m_TaskQueFront];
		m_TaskQueFront = (m_TaskQueFront + 1) % TASK_QUE_SIZE;
		//printf("RemoteAssistant Run msgTag = %d\n",((Msg_t *)m_Buffer)->msgTag);
		switch(((Msg_t*)m_Buffer)->msgTag) {
		case CudaDeviceSetCacheConfigTag:	CudaDeviceSetCacheConfig();	break;
		case CudaGetDevicePropertiesTag:	CudaGetDeviceProperties(); 	break;
		case CudaGetErrorStringTag:	CudaGetErrorString();	break;
		case CudaStreamCreateTag:	CudaStreamCreate(); 	break;
		case CudaStreamDestroyTag:	CudaStreamDestroy();	break;
		case CudaStreamSynchronizeTag:	CudaStreamSynchronize();	break;
		case CudaMallocTag:		CudaMalloc();	break;
		case CudaMemsetTag: CudaMemset(); break;
		case CudaFreeTag:		CudaFree();		break;
		case CudaMemcpyAsyncTag:	CudaMemcpyAsync();	break;
		case CudaConfigureCallTag:	CudaConfigureCall();	break;
		case CudaSetupArgumentTag:	CudaSetupArgument();	break;
		case CudaModuleLoadTag:	CudaModuleLoad(); break;
		case CudaModuleGetFuncTag:	CudaModuleGetFunction();	break;
		case CudaModuleGetSymbolTag:	CudaModuleGetSymbol();	break;
		case CudaLaunchTag:		CudaLaunch();	break;
		case CudaDeviceSynchronizeTag:	CudaDeviceSynchronize();	break;
		case CudaDeviceResetTag:	CudaDeviceReset();	break;
		case CudaDeviceSetLimitTag: CudaDeviceSetLimit();break;
		case CudaEventCreateTag: CudaEventCreate();break;
		case CudaEventDestroyTag: CudaEventDestroy();break;
		case CudaEventRecordTag: CudaEventRecord();break;
		case CudaEventSynchronizeTag: CudaEventSynchronize();break;
		case CudaEventElapsedTimeTag: CudaEventElapsedTime();break;
		case NullTag:	runFlag = false;	break;
		default:;
		}

		sem_post(&m_FreeTaskSlotCount);
	}
	cudaDeviceReset();
}

void msgPrint(int signo)
{
	printf("RemoteAssistant get signal 11 deviceId = %d\n",globalDeviceId);
}

int main(int argc, char *argv[]) {
	signal(11,msgPrint);
	RemoteAssistant remoteAssistant(&argc, &argv);
	remoteAssistant.Run();
	//printf("RemoteAssistant main exit\n");
}
