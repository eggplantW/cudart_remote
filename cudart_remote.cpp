#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <__cudaFatFormat.h>
#include <fatBinaryCtl.h>
#include <fatbinary.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <pthread.h>

#include "Functions.h"
#include "Defines.h"
#include "Types.h"
#include "cudart_remote.h"

#include <unistd.h>	//for sleep
#include <signal.h>//for signal caught

#define GC_GET_LOCAL_FUNC_PTR(func) GC_GetLocalFuncPtr((void *)func, #func)
#define GC_PTX_PATH "."
#define GC_RemoteExecPath "/home/run/cuda-workplace/cudart_remote/src/RemoteAssistant"
#define CU_MODULE_NULL NULL

class GC_InitStruct 
{
	public:
	GC_InitStruct() 
	{
		cudaRemoteInit(NULL, NULL);
	}
	~GC_InitStruct() 
	{
		cudaRemoteFinalize();
	}
};

MPI_Comm GC_DeviceComm;

static int GC_LocalNode;
static int GC_DeviceCount,GC_NodeCount;
static int GC_ThreadCount = 1;
pthread_mutex_t GC_ThreadInfoMutex;

enum GC_DeviceObjectTag { DeviceFunction, DeviceVariable };
typedef struct GC_DeviceFuncInfo 
{
	char* deviceFunc;
	char* deviceName;
	int thread_limit;
	uint3* tid;
	uint3* bid;
	dim3* bDim;
	dim3* gDim;
	int* wSize;
	CUfunction* hfuncList;
	pthread_mutex_t *hfuncListMutex;
	size_t hfuncListSize;
} *GC_DeviceFuncInfo_t;
typedef struct GC_DeviceVarInfo 
{
	char* deviceAddress;
	char* deviceName;
	int ext;
	int size;
	int constant;
	int global;
	void** dptrList;
	pthread_mutex_t *dptrListMutex;
	size_t dptrListSize;
} *GC_DeviceVarInfo_t;
//record the map between host pointer and device function or symbol info handle
typedef struct GC_HostPtrMapName
{
	const void *hostPtr;
	int fatCubinHandleId;
	enum GC_DeviceObjectTag tag;
	void* deviceInfo;
	GC_HostPtrMapName *next;
} *GC_HostPtrMapName_t, *GC_FuncPtrMapName_t, *GC_VarPtrMapName_t;
GC_FuncPtrMapName_t GC_FuncPtrMapNameHeader = NULL;	//header of list of <hostPtr, functionName>
GC_VarPtrMapName_t GC_VarPtrMapNameHeader = NULL;	//header of list of <hostPtr, symbolName>

//store the fatCubinHandle and its module, if the module is null, the fatCubin has not been loaded
#define GC_MODULE_ARRAY_SLICE 8

typedef struct GC_FatCubinHandle
{
	void* fatCubin;
	size_t fatCubinSize;
	CUmodule *moduleArray;
	pthread_mutex_t *moduleArrayMutex;
	size_t moduleArraySize;
} *GC_FatCubinHandle_t;
#define GC_FATCUBIN_ARRAY_SLICE 8
struct GC_FatCubinHandleVector
{
	GC_FatCubinHandle_t *fatCubinHandleArray;
	size_t fatCubinHandleCount;
	size_t fatCubinHandleArraySize;
} GC_FatCubinHandles = { NULL, 0, 0 };

//stream attr recorder
struct StreamAttr 
{
	int streamTag;
	int deviceId;
	std::vector<MPI_Request>lastReq;
	StreamAttr(): streamTag(ThreadCallTag), deviceId(0){lastReq.clear();}
	StreamAttr(int _streamTag, int _deviceId): streamTag(_streamTag), deviceId(_deviceId){lastReq.clear();}
};
struct EventAttr
{
	int deviceId;
	cudaStream_t streamAttach;
	EventAttr(): deviceId(0),streamAttach(NULL){}
	EventAttr(int _deviceId, cudaStream_t _streamAttach): deviceId(_deviceId),streamAttach(_streamAttach){}
};

struct ThreadInfo
{
	int currentDevice;
	int threadId;
	char *cudaErrorString;
	int cudaErrorStringLen;
	cudaError_t cudaLastError;
	ThreadInfo():currentDevice(0),threadId(1),cudaErrorString(NULL),cudaErrorStringLen(0),cudaLastError(cudaSuccess){}
};

int* GC_DeviceList;
bool* GC_DeviceInitFlag;
cudaStream_t* GC_StreamNull;
pthread_mutex_t GC_StreamNullMutex;

std::vector<std::map<cudaStream_t, StreamAttr> >GC_StreamTag;
std::vector<std::map<cudaEvent_t, EventAttr> >GC_EventTag;

pthread_mutex_t GC_StreamTagMutex;
pthread_mutex_t GC_EventTagMutex;



std::map<pthread_t,ThreadInfo> GC_ThreadInfo;

static char* GC_CudaErrorString;

GC_InitStruct GC_Init;

extern "C" 
{
	static __host__ int CUDARTAPI GC_FatCubinRegister(void* fatCubin) 
	{
		if(GC_FatCubinHandles.fatCubinHandleCount == GC_FatCubinHandles.fatCubinHandleArraySize) {
			GC_FatCubinHandles.fatCubinHandleArray = (GC_FatCubinHandle_t *)realloc(GC_FatCubinHandles.fatCubinHandleArray, sizeof(GC_FatCubinHandle_t)*(GC_FatCubinHandles.fatCubinHandleArraySize+GC_FATCUBIN_ARRAY_SLICE));
			GC_FatCubinHandles.fatCubinHandleArraySize += GC_FATCUBIN_ARRAY_SLICE;
		}
		computeFatBinaryFormat_t fatHeader = (computeFatBinaryFormat_t)fatCubin;
		GC_FatCubinHandle_t p = (GC_FatCubinHandle_t)malloc(sizeof(struct GC_FatCubinHandle));
		p->fatCubinSize = fatHeader->fatSize + fatHeader->headerSize;
		p->fatCubin = malloc(p->fatCubinSize);
		memcpy(p->fatCubin, fatCubin, p->fatCubinSize);	//restore fatCubin
		//init module array
		p->moduleArray = (CUmodule *)malloc(sizeof(CUmodule)*(GC_DeviceCount / GC_NodeCount));
		p->moduleArrayMutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * (GC_DeviceCount / GC_NodeCount));
		for(int i = 0; i < (GC_DeviceCount / GC_NodeCount); i++)
		{
			p->moduleArray[i] = CU_MODULE_NULL;
			pthread_mutex_init(&(p->moduleArrayMutex[i]),NULL);
		}
		p->moduleArraySize = (GC_DeviceCount / GC_NodeCount);
		GC_FatCubinHandles.fatCubinHandleArray[GC_FatCubinHandles.fatCubinHandleCount] = p;
		return GC_FatCubinHandles.fatCubinHandleCount++;
	}

	static __host__ void CUDARTAPI GC_ModuleRegister(int fatCubinHandleId, CUmodule module, int device) 
	{
		GC_FatCubinHandle_t _fatCubinHandle = GC_FatCubinHandles.fatCubinHandleArray[fatCubinHandleId];
		_fatCubinHandle->moduleArray[device] = module;
	}

	static __host__ void CUDARTAPI GC_FatCubinUnRegisterAll() 
	{
		int fatCubinCount = GC_FatCubinHandles.fatCubinHandleCount;
		for(int i = 0; i < fatCubinCount; i++) {
			GC_FatCubinHandle_t p = GC_FatCubinHandles.fatCubinHandleArray[i];
			free(p->fatCubin);
			free(p->moduleArray);
			for(int j = 0;j < (GC_DeviceCount / GC_NodeCount);j ++)
				pthread_mutex_destroy(&(p->moduleArrayMutex[i]));
			free(p->moduleArrayMutex);
			free(p);
		}
		free(GC_FatCubinHandles.fatCubinHandleArray);
		GC_FatCubinHandles.fatCubinHandleArray = NULL;
		GC_FatCubinHandles.fatCubinHandleArraySize = 0;
		GC_FatCubinHandles.fatCubinHandleCount = 0;
	}

	static __host__ GC_DeviceFuncInfo_t CUDARTAPI GC_StoreFuncInfo(
			char    *deviceFun,
	  const char    *deviceName,
			int      thread_limit,
			uint3   *tid,
			uint3   *bid,
			dim3    *bDim,
			dim3    *gDim,
			int     *wSize) 
	{
		size_t deviceNameLen = strlen(deviceName);
		GC_DeviceFuncInfo_t p_DeviceInfo = (GC_DeviceFuncInfo_t)malloc(sizeof(struct GC_DeviceFuncInfo));
		p_DeviceInfo->deviceName = (char*)malloc(deviceNameLen + 1);
		strcpy(p_DeviceInfo->deviceName, deviceName);
		p_DeviceInfo->hfuncList = (CUfunction*)malloc((GC_DeviceCount / GC_NodeCount) * sizeof(CUfunction));
		p_DeviceInfo->hfuncListMutex = (pthread_mutex_t*)malloc((GC_DeviceCount / GC_NodeCount) * sizeof(pthread_mutex_t));
		p_DeviceInfo->hfuncListSize = (GC_DeviceCount / GC_NodeCount);
		for(int i = 0; i < p_DeviceInfo->hfuncListSize; i++)
		{
			p_DeviceInfo->hfuncList[i] = NULL;
			pthread_mutex_init(&(p_DeviceInfo->hfuncListMutex[i]),NULL);
		}
		return p_DeviceInfo;
	}

	static __host__ void CUDARTAPI GC_InsertCUfunction(GC_DeviceFuncInfo_t p_DeviceFuncInfo, CUfunction hfunc, int deviceId) 
	{
		if(p_DeviceFuncInfo == NULL)
			return;
		p_DeviceFuncInfo->hfuncList[deviceId] = hfunc;
	}

	static __host__ void CUDARTAPI GC_DelFuncInfo(GC_DeviceFuncInfo_t p_DeviceFuncInfo) 
	{
		if( p_DeviceFuncInfo == NULL )
			return;
		if(p_DeviceFuncInfo->deviceName != NULL)
			free(p_DeviceFuncInfo->deviceName);
		if(p_DeviceFuncInfo->hfuncList != NULL)
			free(p_DeviceFuncInfo->hfuncList);
		for(int i = 0; i < p_DeviceFuncInfo->hfuncListSize; i++)
		{
			pthread_mutex_destroy(&(p_DeviceFuncInfo->hfuncListMutex[i]));
		}
		free(p_DeviceFuncInfo->hfuncListMutex);
		free(p_DeviceFuncInfo);
	}

	static __host__ GC_DeviceVarInfo_t CUDARTAPI GC_StoreVarInfo(
			char  *deviceAddress,
	  const char  *deviceName,
			int    ext,
			int    size,
			int    constant,
			int    global
	) 
	{
		size_t deviceNameLen = strlen(deviceName);
		GC_DeviceVarInfo_t p_DeviceInfo = (GC_DeviceVarInfo_t)malloc(sizeof(struct GC_DeviceVarInfo));
		p_DeviceInfo->deviceName = (char*)malloc(deviceNameLen + 1);
		strcpy(p_DeviceInfo->deviceName, deviceName);
		p_DeviceInfo->ext = ext;
		p_DeviceInfo->size = size;
		p_DeviceInfo->constant = constant;
		p_DeviceInfo->global = global;
		p_DeviceInfo->dptrList = (void**)malloc((GC_DeviceCount / GC_NodeCount) * sizeof(void*));
		p_DeviceInfo->dptrListMutex = (pthread_mutex_t *)malloc((GC_DeviceCount / GC_NodeCount) * sizeof(pthread_mutex_t));
		p_DeviceInfo->dptrListSize = (GC_DeviceCount / GC_NodeCount);
		for(int i = 0; i < (GC_DeviceCount / GC_NodeCount); i++)
		{
			p_DeviceInfo->dptrList[i] = NULL;
			pthread_mutex_init(&(p_DeviceInfo->dptrListMutex[i]),NULL);
		}
		return p_DeviceInfo;
	}

	static __host__ void CUDARTAPI GC_InsertDptr(GC_DeviceVarInfo_t p_DeviceVarInfo, void* dptr, int deviceId)
	{
		if(p_DeviceVarInfo == NULL)
			return;
		p_DeviceVarInfo->dptrList[deviceId] = dptr;
	}

	static __host__ void CUDARTAPI GC_DelVarInfo(GC_DeviceVarInfo_t p_DeviceVarInfo) 
	{
		if(p_DeviceVarInfo == NULL)
			return;
		if(p_DeviceVarInfo->deviceName != NULL)
			free(p_DeviceVarInfo->deviceName);
		if(p_DeviceVarInfo->dptrList != NULL)
			free(p_DeviceVarInfo->dptrList);
		for(int i = 0; i < (GC_DeviceCount / GC_NodeCount); i++)
		{
			pthread_mutex_destroy(&(p_DeviceVarInfo->dptrListMutex[i]));
		}
		free(&(p_DeviceVarInfo->dptrListMutex));
		free(p_DeviceVarInfo);
	}

	static __host__ GC_HostPtrMapName_t CUDARTAPI GC_HostPtrMapNameInsert(
		GC_HostPtrMapName_t header, const void *hostPtr,
		int fatCubinHandleId, enum GC_DeviceObjectTag tag, void* deviceInfo)
	{
		GC_HostPtrMapName_t p = header, s = NULL;
		while(p != NULL) 
		{
			if(p->hostPtr == hostPtr) 
			{
				p->fatCubinHandleId = fatCubinHandleId;
				p->tag = tag;
				p->deviceInfo = deviceInfo;
				break;
			}
			s = p;
			p = p->next;
		}
		if(p == NULL) 
		{
			p = (GC_HostPtrMapName_t)malloc(sizeof(struct GC_HostPtrMapName));
			p->hostPtr = hostPtr;
			p->fatCubinHandleId = fatCubinHandleId;
			p->tag = tag;
			p->deviceInfo = deviceInfo;
			p->next = NULL;
			if(s == NULL)
				header = p;
			else
				s->next = p;
		}
		return header;
	}

	static __host__ void CUDAAPI GC_HostPtrMapNameClear( GC_HostPtrMapName_t header ) 
	{
		GC_HostPtrMapName_t p = header, s = NULL;
		while( p != NULL) {
			s = p;
			p = p->next;
			if(s->deviceInfo) 
			{
				switch(s->tag) 
				{
					case DeviceFunction:	GC_DelFuncInfo((GC_DeviceFuncInfo_t)s->deviceInfo); break;
					case DeviceVariable:	GC_DelVarInfo((GC_DeviceVarInfo_t)s->deviceInfo); break;
					default:;
				}
			}
			free(s);
		}
	}

	void** CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) 
	{
		//analysis the fatCubin to get its size
		__fatBinC_Wrapper_t *fatbinWrapper = (__fatBinC_Wrapper_t *)fatCubin;
		unsigned long long handleId = GC_FatCubinRegister((void*)fatbinWrapper->data);
		return (void**)handleId;
	}

	void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
	}

	void CUDARTAPI __cudaRegisterFunction(
			void   **fatCubinHandle,
	  const char    *hostFun,
			char    *deviceFun,
	  const char    *deviceName,
			int      thread_limit,
			uint3   *tid,
			uint3   *bid,
			dim3    *bDim,
			dim3    *gDim,
			int     *wSize) 
	{
		GC_DeviceFuncInfo_t deviceFuncInfo = GC_StoreFuncInfo(deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
		GC_FuncPtrMapNameHeader = GC_HostPtrMapNameInsert(GC_FuncPtrMapNameHeader, hostFun, (unsigned long long)fatCubinHandle, DeviceFunction, deviceFuncInfo);


	}

	void CUDARTAPI __cudaRegisterVar(
			void **fatCubinHandle,
			char  *hostVar,
			char  *deviceAddress,
	  const char  *deviceName,
			int    ext,
			int    size,
			int    constant,
			int    global) 
	{
		GC_DeviceVarInfo_t deviceVarInfo = GC_StoreVarInfo(deviceAddress, deviceName, ext, size, constant, global);
		GC_VarPtrMapNameHeader = GC_HostPtrMapNameInsert(GC_VarPtrMapNameHeader, hostVar, (unsigned long long)fatCubinHandle, DeviceVariable, deviceVarInfo);
	}

	char CUDARTAPI __cudaInitModule(void **fatCubinHandle) 
	{
		return 0;
	}

	static __host__ void CUDARTAPI GC_DeviceConfig() 
	{
		int i;
		char *deviceList = getenv("GC_DEVICE_LIST");
		if(deviceList == NULL) 
		{
			std::cerr << "cannot find GC_DEVICE_NUM_LIST" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		char device_num_str[4];
		char *deviceList_t = deviceList;
		GC_DeviceCount = 0;
		while( sscanf(deviceList_t, "%s", device_num_str) > 0 ) 
		{
			deviceList_t += strlen(device_num_str);
			while(*deviceList_t == ' ' || *deviceList_t == '\t')
				deviceList_t++;
			GC_DeviceCount++;
		}
		if(GC_DeviceCount % GC_NodeCount != 0)
		{
			std::cerr << "DeviceNum % TaskNum != 0" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
		
		GC_DeviceInitFlag = (bool *)malloc(sizeof(bool) * (GC_DeviceCount / GC_NodeCount));
		for(i = 0; i < GC_DeviceCount / GC_NodeCount; i++)
			GC_DeviceInitFlag[i] = false;		
		GC_StreamNull = (cudaStream_t *)malloc(sizeof(cudaStream_t) * (GC_DeviceCount / GC_NodeCount));
		GC_DeviceList = (int *)malloc(sizeof(int) * (GC_DeviceCount / GC_NodeCount));
		for(i = 0;i < GC_DeviceCount / GC_NodeCount; i++)
			GC_DeviceList[i] = GC_DeviceCount / GC_NodeCount * GC_LocalNode + i;
		
		GC_StreamTag.resize(GC_DeviceCount / GC_NodeCount);
		GC_EventTag.resize(GC_DeviceCount / GC_NodeCount);
	}

	void msgPrint(int signo)
	{
		//printf("cudart get signal 11 pid = %x\n",pthread_self());
		exit(1);
	}

	__host__ void CUDARTAPI cudaRemoteInit(int *argc, char ***argv) 
	{
		signal(11,msgPrint);
		int mpiInitFlag = 0;
		MPI_Initialized(&mpiInitFlag);
		int provided;
		if(!mpiInitFlag)
		{
			mpi_error( MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided) );
			if(provided != MPI_THREAD_MULTIPLE)
				printf("mpi doesn't support multiple thread\n");
			
				//printf("mpi support multiple thread\n");
		}
		mpi_error( MPI_Comm_rank(MPI_COMM_WORLD, &GC_LocalNode) );
		mpi_error( MPI_Comm_size(MPI_COMM_WORLD, &GC_NodeCount) );
		GC_DeviceConfig();
		MPI_Info info = MPI_INFO_NULL;
		char *deviceHosts = getenv("GC_DEVICE_HOSTS");
		MPI_Info_create(&info);
		MPI_Info_set(info,"hostfile",deviceHosts);
		mpi_error( MPI_Comm_spawn(GC_RemoteExecPath, MPI_ARGV_NULL, GC_DeviceCount, info, 0, MPI_COMM_WORLD, &GC_DeviceComm, MPI_ERRCODES_IGNORE) );

		pthread_mutex_init(&GC_ThreadInfoMutex,NULL);
		pthread_mutex_init(&GC_StreamNullMutex,NULL);
		pthread_mutex_init(&GC_StreamTagMutex,NULL);
		pthread_mutex_init(&GC_EventTagMutex,NULL);
	}

	__host__ void CUDARTAPI cudaRemoteFinalize() 
	{
		/*
		GC_HostPtrMapNameClear(GC_FuncPtrMapNameHeader);
		GC_HostPtrMapNameClear(GC_VarPtrMapNameHeader);
		GC_FatCubinUnRegisterAll();
		
		pthread_mutex_destroy(&GC_ThreadInfoMutex); 
		pthread_mutex_destroy(&GC_StreamNullMutex); 
		pthread_mutex_destroy(&GC_StreamTagMutex); 
		
		free(GC_DeviceInitFlag);
		free(GC_StreamNull);
		free(GC_DeviceList);
		*/
		Msg_t msg(NullTag);

		for(int i = 0; i < (GC_DeviceCount / GC_NodeCount); i++) 
		{
			//printf("send fini %d\n",i);
			mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[i], NullTag, GC_DeviceComm) );
		}
		/*  int flag;
		for(int i = 0; i < (GC_DeviceCount / GC_NodeCount); i++) 
		{
			mpi_error( MPI_Recv(&flag, sizeof(int), MPI_BYTE, GC_DeviceList[i], NullTag, GC_DeviceComm,MPI_STATUS_IGNORE) );
		}
		*/
		//mpi_error( MPI_Recv(NULL, 0, MPI_BYTE, 0, 0, GC_DeviceComm, MPI_STATUS_IGNORE) );
		//printf("0");
		//sleep(10);
		//printf("cudart before mpi_finalize\n");
		MPI_Finalize();
		//exit(0);
		//printf("cudart after mpi_finalize\n");
		//sleep(100);
	}

	__host__ void CUDARTAPI threadInfoTest(pthread_t pid)
	{
		std::map<pthread_t,ThreadInfo>::iterator it = GC_ThreadInfo.find(pid);
		if(it == GC_ThreadInfo.end())
		{
			ThreadInfo info = ThreadInfo();
			info.currentDevice = 0;
			info.cudaLastError = cudaSuccess;			
			info.threadId = GC_ThreadCount;
			GC_ThreadCount ++;			
			info.cudaErrorString = new char[256];
			info.cudaErrorStringLen = 256;
			GC_ThreadInfo[pid] = info;
		}
		return ;
	}

	static __host__ __cudart_builtin__ cudaError_t CUDARTAPI GC_CreateStream(cudaStream_t *pStream, enum GC_StreamFlag flag) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);	
		threadInfoTest(pid);		
		ThreadInfo info = GC_ThreadInfo[pid];
		int dest = GC_DeviceList[info.currentDevice];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		CudaStreamCreateMsg_t msg(flag,threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		CudaStreamCreateAckMsg_t amsg;

		mpi_error( MPI_Recv(&amsg, sizeof(amsg), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		if(amsg.status == cudaSuccess) {
			*pStream = amsg.stream;
			pthread_mutex_lock(&GC_StreamTagMutex);
			GC_StreamTag[info.currentDevice][amsg.stream] = StreamAttr(amsg.streamTag,info.currentDevice);
			pthread_mutex_unlock(&GC_StreamTagMutex);
		}
		else
			*pStream = NULL;
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = amsg.status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		return amsg.status;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		threadInfoTest(pid);
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		*count = GC_DeviceCount / GC_NodeCount;
		return cudaSuccess;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		*device = info.currentDevice;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		return cudaSuccess;
	}

	__host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int dest = GC_DeviceList[info.currentDevice];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		CudaDeviceSetCacheConfigMsg_t msg(cacheConfig,threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );

		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		return cudaErr;
	}

	__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) 
	{
		//printf("cudaSetDevice device = %d\n",device);
		if(device < 0 || device >= (GC_DeviceCount / GC_NodeCount))
			return cudaErrorInvalidDevice;
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		threadInfoTest(pid);
		int tmpDevice = GC_ThreadInfo[pid].currentDevice;
		GC_ThreadInfo[pid].currentDevice = device;
		int threadId = GC_ThreadInfo[pid].threadId;
		int currentDevice = GC_ThreadInfo[pid].currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		if(GC_DeviceInitFlag[device] == true)
		{
			//printf("cudaSetDevice at 1:pid = %d,threadId = %d,currentDevice = %d,device = %d\n",pid,threadId,currentDevice,device);
			pthread_mutex_lock(&GC_ThreadInfoMutex);
			GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
			pthread_mutex_unlock(&GC_ThreadInfoMutex);
			return cudaSuccess;
		}
		pthread_mutex_lock(&GC_StreamNullMutex);
		if(GC_DeviceInitFlag[device] == true)
		{
			//printf("cudaSetDevice at 2:pid = %x,threadId = %d,currentDevice = %d,device = %d\n",pid,threadId,currentDevice,device);
			pthread_mutex_lock(&GC_ThreadInfoMutex);
			GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
			pthread_mutex_unlock(&GC_ThreadInfoMutex);
			pthread_mutex_unlock(&GC_StreamNullMutex);
			return cudaSuccess;
		}
		cudaStream_t stream_null;
		cudaError_t cudaErr = GC_CreateStream(&stream_null, NullStreamFlag);
		if(cudaErr != cudaSuccess)
		{
			//printf("cudaSetDevice at 3:pid = %x,threadId = %d,currentDevice = %d,device = %d\n",pid,threadId,currentDevice,device);
			pthread_mutex_lock(&GC_ThreadInfoMutex);
			GC_ThreadInfo[pid].cudaLastError = cudaErr;
			GC_ThreadInfo[pid].currentDevice = tmpDevice;
			pthread_mutex_unlock(&GC_ThreadInfoMutex);
			pthread_mutex_unlock(&GC_StreamNullMutex);
			return cudaErr;
		}
		GC_StreamNull[device] = stream_null;
		//printf("cudaSetDevice pid = %x,threadId = %d,device = %d,currentDevice = %d,defaultStream = %x,stream_null = %x\n",pid,threadId,device,currentDevice,GC_StreamNull[device],stream_null);
		GC_DeviceInitFlag[device] = true;
		pthread_mutex_unlock(&GC_StreamNullMutex);
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		return cudaSuccess;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) 
	{
		//printf("cudaGetDevicePropertites\n");
		if(device < 0 || device > GC_DeviceCount)
			return cudaErrorInvalidDevice;
		pthread_t pid= pthread_self();

		pthread_mutex_lock(&GC_ThreadInfoMutex);
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		CudaGetDevicePropertiesMsg_t msg(threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		CudaGetDevicePropertiesAckMsg_t amsg;
		mpi_error( MPI_Recv(&amsg, sizeof(amsg), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		if(amsg.cudaStat == cudaSuccess)
			*prop = amsg.prop;

		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = amsg.cudaStat;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		return amsg.cudaStat;
	}

	__host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) 
	{
		pthread_t pid= pthread_self();

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int dest = GC_DeviceList[info.currentDevice];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		CudaGetErrorStringMsg_t msg(error,threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		MPI_Status stat;
		mpi_error( MPI_Probe(dest, threadId << 16, GC_DeviceComm, &stat) );
		int errStringLen;
		mpi_error( MPI_Get_count(&stat, MPI_BYTE, &errStringLen) );
		if(errStringLen > info.cudaErrorStringLen) {
			delete[]  info.cudaErrorString;
			info.cudaErrorString = new char[errStringLen];
			info.cudaErrorStringLen = errStringLen;
		}
		mpi_error( MPI_Recv(info.cudaErrorString, errStringLen, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, GC_DeviceComm, MPI_STATUS_IGNORE) );

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;

		return info.cudaErrorString;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) 
	{
		if(flags == cudaStreamDefault)
			return GC_CreateStream(pStream, DefaultStreamFlag);
		else if(flags == cudaStreamNonBlocking)
			return GC_CreateStream(pStream, NonblockingStreamFlag);
		else
			return cudaErrorInvalidValue;
	}

	__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) 
	{
		return GC_CreateStream(pStream, DefaultStreamFlag);
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) 
	{
		if(stream == NULL) 
		{
		//stream = GC_StreamNull[GC_CurrentDevice];
		return cudaSuccess;
		}
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		int deviceId;
		pthread_mutex_lock(&GC_StreamTagMutex);
		std::map<cudaStream_t, StreamAttr>::iterator it = GC_StreamTag[device].find(stream);
		if(it == GC_StreamTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidResourceHandle;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_StreamTagMutex);
			return cudaErrorInvalidResourceHandle;
		}
		deviceId = it->second.deviceId;
		GC_StreamTag[device].erase(stream);
		pthread_mutex_unlock(&GC_StreamTagMutex);
		CudaStreamDestroyMsg_t msg(stream,threadId);
		mpi_error( MPI_Send(&msg, sizeof(CudaStreamDestroyMsg_t), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		cudaError_t status;
		//printf("host at |%d|,buf = %x\n",__LINE__,&status);
		//printf("host at |%d|,buf = %x,count = %d,src = %d,tag = %d,comm = %x\n",__LINE__,&status,sizeof(cudaError_t),GC_DeviceList[device],threadId << 16,GC_DeviceComm);
		mpi_error( MPI_Recv(&status, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		return status;
	}

	__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		if(stream == NULL)
			stream = GC_StreamNull[device];
		//printf("cudaStreamSynchronize:pid = %x,threadId = %d,currentDevice = %d,stream = %x\n",pid,threadId,device,stream);
		CudaStreamSynchronizeMsg_t msg(stream,threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		cudaError_t status;
		mpi_error( MPI_Recv(&status, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return status;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);	
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		CudaMallocMsg_t msg(size,threadId);
		//printf("cudaMalloc before MPI_Send:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Send(&msg, sizeof(CudaMallocMsg_t), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		CudaMallocAckMsg_t amsg;
		//printf("cudaMalloc before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Recv(&amsg, sizeof(CudaMallocAckMsg_t), MPI_BYTE, GC_DeviceList[device],info.threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		int64_t tmpP = (int64_t)amsg.dptr;
		//printf("threadId = %d,tmpP = %lx device = %d device48 = %lx\n",threadId,tmpP,device,(long)device << 48);	
		tmpP = (tmpP & 0x0000ffffffffffff) | 0x8000000000000000 | (((long)device) << 48) ;
		//printf("threadId = %d,tmpP1 = %lx \n",threadId,tmpP);	
		*devPtr = (void*) tmpP;
		//printf("threadId = %d,dptr = %lx,tmpP = %lx \n",threadId,amsg.dptr,*devPtr);	
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = amsg.status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return amsg.status;
	}
	//to be changed
	__host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		*pHost = malloc(size);
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaSuccess;
	}
	//to be changed
	__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		*ptr = malloc(size);
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaSuccess;
	}
	//added on 	20160908
	__host__ __cudart_builtin__ cudaError_t cudaMemset(void *devPtr,int value, size_t size)
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		cudaError_t cudaResu;
		int64_t tmpP = (int64_t)devPtr;
		devPtr = (void *)(0x0000ffffffffffff & tmpP);
		device = (tmpP << 1) >> 49;
		CudaMemsetMsg_t msg(devPtr,value,size,threadId);
		//printf("cudaMemset before MPI_Send:pid = %x,threadId = %d,currentDevice = %d devPtr = %x,value = %d,size = %d\n",pid,threadId,device,devPtr,value,size);
		mpi_error( MPI_Send(&msg, sizeof(CudaMemsetMsg_t), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		MPI_Status status;
		//printf("cudaMemset before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Recv(&cudaResu, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, &status) );
		//printf("cudaMemset after MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaResu;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaResu;
	}
	__host__ __cudart_builtin__ cudaError_t cudaFree(void *devPtr)
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		cudaError_t cudaResu;
		int64_t tmpP = (int64_t)devPtr;
		devPtr = (void *)(0x0000ffffffffffff & tmpP);
		device = (tmpP << 1) >> 49;
		CudaFreeMsg_t msg(devPtr,threadId);
		//printf("cudaFree before MPI_Send:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Send(&msg, sizeof(CudaFreeMsg_t), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		MPI_Status status;
		//printf("cudaFree before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Recv(&cudaResu, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, &status) );

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaResu;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaResu;
	}

	__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		if(ptr != NULL)
		{
			free(ptr);
			GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		}
		else
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInitializationError;
		cudaError_t err = GC_ThreadInfo[pid].cudaLastError;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return err;
	}

	__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		//printf("cudaMemcpy:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		if(!GC_DeviceInitFlag[device]) 
		{
			cudaError_t stat = cudaSetDevice(device);
			if(stat != cudaSuccess) 
			{
				pthread_mutex_lock(&GC_ThreadInfoMutex)	;
				GC_ThreadInfo[pid].cudaLastError = stat;
				pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
				return stat;
			}
		}
		cudaError_t cudaErr = cudaMemcpyAsync(dst, src, count, kind, 0);
		if(cudaErr != cudaSuccess)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErr;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErr;
		}
		cudaErr = cudaStreamSynchronize(0);
		//printf("mpi wait gc %d\n",GC_ThreadInfo[pid].lastReq);
		//MPI_Wait(GC_ThreadInfo[pid].lastReq,MPI_STATUS_IGNORE);
		//printf("mpi wait gc %d fini\n",GC_ThreadInfo[pid].lastReq);
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		return cudaErr;
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;

		if(stream == NULL)
		{
			cudaSetDevice(device);
			stream = GC_StreamNull[device];
		}
		//printf("cudaMemcpyAsync:pid = %x,threadId = %d,currentDevice = %d,stream = %x,dst = %x\n",pid,threadId,device,stream,dst);

		pthread_mutex_lock(&GC_StreamTagMutex);
		std::map<cudaStream_t,StreamAttr>::iterator streamIt = GC_StreamTag[device].find(stream) ;
		if(streamIt == GC_StreamTag[device].end())
		{
			//printf("host memcpyasync didn't find stream:%x,pid = %x\n",stream,pid);
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			pthread_mutex_unlock(&GC_StreamTagMutex);
			return cudaErrorInvalidValue;
		}
		int streamTag = streamIt->second.streamTag;
		pthread_mutex_unlock(&GC_StreamTagMutex);

		void* _src = const_cast<void*>(src);
		int64_t dstP = (int64_t)dst;
		int64_t srcP = (int64_t)_src;
		//printf("threadId = %d,dstP = %lx,srcP = %lx, dstP63 = %lx, srcP63 = %lx\n",threadId,dstP,srcP,dstP >> 63,srcP >> 63);
		int sum = 0;
		if(kind == cudaMemcpyHostToDevice) {		
			if(dstP >> 63 == 0 || srcP >> 63 != 0)
			{
				pthread_mutex_lock(&GC_ThreadInfoMutex)	;
				GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidDevicePointer;
				pthread_mutex_unlock(&GC_ThreadInfoMutex);
				return cudaErrorInvalidDevicePointer;
			}
			dst = (void *)(dstP & 0x0000ffffffffffff);
			CudaMemcpyAsyncMsg_t msg(dst, count, kind, stream, threadId);
			mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
			
			MPI_Send(_src,count,MPI_BYTE,GC_DeviceList[device],threadId<<16|streamTag,GC_DeviceComm);
		}
		else if(kind == cudaMemcpyDeviceToHost) {
			if(srcP >> 63 == 0 || dstP >> 63 != 0)
			{
				pthread_mutex_lock(&GC_ThreadInfoMutex)	;
				GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidDevicePointer;
				pthread_mutex_unlock(&GC_ThreadInfoMutex);
				return cudaErrorInvalidDevicePointer;
			}
			_src = (void *)(srcP & 0x0000ffffffffffff);
			CudaMemcpyAsyncMsg_t msg(_src, count, kind, stream, threadId);
			mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
			//printf("cudart memcpy device to host at %d %d\n",__LINE__,sum);
			
			MPI_Recv(dst,count,MPI_BYTE,GC_DeviceList[device],threadId << 16|streamTag,GC_DeviceComm,MPI_STATUS_IGNORE);
		}
		else
		{	
			if(srcP >> 63 == 0 || dstP >> 63 == 0)
			{
				pthread_mutex_lock(&GC_ThreadInfoMutex)	;
				GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidDevicePointer;
				pthread_mutex_unlock(&GC_ThreadInfoMutex);
				return cudaErrorInvalidDevicePointer;
			}
			int srcDevice = (int)((srcP << 1) >> 49);
			_src = (void *)(srcP & 0x0000ffffffffffff);
			int dstDevice = (int)((dstP << 1) >> 49);
			dst = (void *)(dstP & 0x0000ffffffffffff);
			void *tmpBuf = malloc(count);
			cudaSetDevice(srcDevice);
			cudaStream_t srcStream = GC_StreamNull[srcDevice];
			CudaMemcpyAsyncMsg_t msg1(_src, count, cudaMemcpyDeviceToHost, srcStream, threadId);
			mpi_error( MPI_Send(&msg1, sizeof(msg1), MPI_BYTE, GC_DeviceList[srcDevice], NullTag, GC_DeviceComm) );
			MPI_Recv(tmpBuf,count,MPI_BYTE,GC_DeviceList[srcDevice],threadId << 16|1,GC_DeviceComm,MPI_STATUS_IGNORE);
			cudaSetDevice(dstDevice);
			cudaStream_t tmpStream = GC_StreamNull[dstDevice];
			CudaMemcpyAsyncMsg_t msg2(dst, count, cudaMemcpyHostToDevice, tmpStream, threadId);
			mpi_error( MPI_Send(&msg2, sizeof(msg2), MPI_BYTE, GC_DeviceList[dstDevice], NullTag, GC_DeviceComm) );
			MPI_Send(tmpBuf,count,MPI_BYTE,GC_DeviceList[dstDevice],threadId<<16|1,GC_DeviceComm);
			cudaSetDevice(srcDevice);
		}
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		return cudaSuccess;
	}

	__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		if(stream == NULL) {
			if(!GC_DeviceInitFlag[device]) 
			{
				cudaError_t stat = cudaSetDevice(device);
				if(stat != cudaSuccess) 
				{
					pthread_mutex_lock(&GC_ThreadInfoMutex)	;
					GC_ThreadInfo[pid].cudaLastError = stat;
					pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
					return stat;
				}
			}
			stream = GC_StreamNull[device];
		}
		CudaConfigureCallMsg_t msg(gridDim, blockDim, sharedMem, stream, threadId);
		//printf("cudaConfigureCall:pid = %x,threadId = %d,currentDevice = %d,stream = %x\n",pid,threadId,device,stream);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaSuccess;
	}

	__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		CudaSetupArgumentMsg_t msg(size, offset, threadId);
		//printf("cudaSetupArgument:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		void* _arg =( const_cast<void*>(arg) );
		int64_t* argInt = (int64_t *)_arg;
		int64_t tmpP = *argInt;
		if(tmpP >> 63 != 0)
		{
			*argInt = (tmpP & 0x0000ffffffffffff);
			_arg = (void *)argInt;
		}
		
		//printf("cudaSetupArgument:pid = %x,threadId = %d,arg = %lx,_arg = %lx,*arg = %lx,size = %d\n",pid,threadId,arg,_arg,*argInt,size);
		mpi_error( MPI_Send(_arg, size, MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm) );

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;

		return cudaSuccess;
	}

	static __host__ CUmodule CUDARTAPI GC_ModuleLoad(int fatHandleId, int device, pthread_t pid) 
	{
		if(fatHandleId < 0 || fatHandleId >= GC_FatCubinHandles.fatCubinHandleCount)
			return CU_MODULE_NULL;
		GC_FatCubinHandle_t fatHandle = GC_FatCubinHandles.fatCubinHandleArray[fatHandleId];

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		if(fatHandle->moduleArray[device] == CU_MODULE_NULL) 
		{
			pthread_mutex_lock(&(fatHandle->moduleArrayMutex[device]));
			if(fatHandle->moduleArray[device] == CU_MODULE_NULL)
			{
				CudaModuleLoadMsg_t msg(threadId);
				//printf("GC_ModuleLoad before MPI_Send:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
				mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
				mpi_error( MPI_Send(fatHandle->fatCubin, fatHandle->fatCubinSize, MPI_BYTE, GC_DeviceList[device],threadId << 16, GC_DeviceComm) );
				CUmodule cuModule = CU_MODULE_NULL;
				//printf("GC_ModuleLoad before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
				mpi_error( MPI_Recv(&cuModule, sizeof(CUmodule), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
				GC_ModuleRegister(fatHandleId, cuModule, device);
			}
			pthread_mutex_unlock(&(fatHandle->moduleArrayMutex[device]));
		}
		return fatHandle->moduleArray[device];
	}

	static __host__ CUfunction CUDARTAPI GC_ModuleGetFunction(GC_DeviceFuncInfo_t p_DeviceFuncInfo, CUmodule module, int device, pthread_t pid) 
	{
		CUfunction hfunc;
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		if(p_DeviceFuncInfo->hfuncList[device] == NULL) 
		{
			pthread_mutex_lock(&(p_DeviceFuncInfo->hfuncListMutex[device]));
			if(p_DeviceFuncInfo->hfuncList[device] == NULL)
			{
				CudaModuleGetFuncMsg_t msg(module,threadId);
				//printf("GC_ModuleGetFunction before MPI_Send:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
				mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
				int nameStrLen = strlen(p_DeviceFuncInfo->deviceName);
				mpi_error( MPI_Send(p_DeviceFuncInfo->deviceName, nameStrLen + 1, MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm) );
				//printf("GC_ModuleGetFunction before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
				mpi_error( MPI_Recv(&hfunc, sizeof(hfunc), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
				GC_InsertCUfunction(p_DeviceFuncInfo, hfunc, device);
			}
			pthread_mutex_unlock(&(p_DeviceFuncInfo->hfuncListMutex[device]));
		}
		return p_DeviceFuncInfo->hfuncList[device];
	}

	__host__ cudaError_t CUDARTAPI cudaLaunch(const void *func) 
	{
		pthread_t pid= pthread_self();

		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		GC_FuncPtrMapName_t p = GC_FuncPtrMapNameHeader;
		while(p != NULL) {
			if(p->hostPtr == func)
				break;
			p = p->next;
		}
		if(p == NULL) {
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidDeviceFunction;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidDeviceFunction;
		}
		GC_DeviceFuncInfo_t funcInfo = (GC_DeviceFuncInfo_t)(p->deviceInfo);
		CUmodule corModule = GC_ModuleLoad(p->fatCubinHandleId, device, pid);
		CUfunction hfunc = GC_ModuleGetFunction(funcInfo, corModule, device, pid);
		CudaLaunchMsg_t msg(corModule, hfunc,threadId);
		//printf("cudaLaunch before MPI_Send:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		//printf("cudaLaunch before MPI_Recv:pid = %x,threadId = %d,currentDevice = %d\n",pid,threadId,device);
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		return cudaErr;
	}

	static __host__ void* CUDARTAPI GC_ModuleGetSymbol(GC_DeviceVarInfo_t p_DeviceVarInfo, CUmodule module, int device, pthread_t pid) {
		void* dptr;
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		if( p_DeviceVarInfo->dptrList[device] == NULL) 
		{
			pthread_mutex_lock(&(p_DeviceVarInfo->dptrListMutex[device]));
			if(p_DeviceVarInfo->dptrList[device] == NULL)
			{
				CudaModuleGetSymbolMsg_t msg(module,threadId);
				mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
				int nameStrLen = strlen(p_DeviceVarInfo->deviceName);
				mpi_error( MPI_Send(p_DeviceVarInfo->deviceName, nameStrLen + 1, MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm) );
				mpi_error( MPI_Recv(&dptr, sizeof(dptr), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
				GC_InsertDptr(p_DeviceVarInfo, dptr, device);
			}
			pthread_mutex_unlock(&(p_DeviceVarInfo->dptrListMutex[device]));
		}
		return p_DeviceVarInfo->dptrList[device];
	}

	__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) 
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		GC_VarPtrMapName_t p = GC_VarPtrMapNameHeader;
		while(p != NULL) {
			if(p->hostPtr == symbol)
				break;
			p = p->next;
		}
		if(p == NULL)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidSymbol;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidSymbol;
		}
		GC_DeviceVarInfo_t varInfo = (GC_DeviceVarInfo_t)(p->deviceInfo);
		CUmodule corModule = GC_ModuleLoad(p->fatCubinHandleId, device, pid);
		void* dptr = GC_ModuleGetSymbol(varInfo, corModule, device, pid);
		if(offset + count > varInfo->size)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidValue;
		}
		if(kind != cudaMemcpyHostToDevice)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidMemcpyDirection;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidMemcpyDirection;
		}
		dptr = (char*)dptr + offset;
		return cudaMemcpyAsync(dptr, src, count, kind, stream);
	}

	__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		GC_VarPtrMapName_t p = GC_VarPtrMapNameHeader;
		while(p != NULL) {
			if(p->hostPtr == symbol)
				break;
			p = p->next;
		}
		if(p == NULL)
		{	
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidSymbol;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidSymbol;
		}
		GC_DeviceVarInfo_t varInfo = (GC_DeviceVarInfo_t)(p->deviceInfo);
		CUmodule corModule = GC_ModuleLoad(p->fatCubinHandleId, device, pid);
		void* dptr = GC_ModuleGetSymbol(varInfo, corModule, device, pid);
		if(offset + count > varInfo->size)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidValue;
		}
		if(kind != cudaMemcpyDeviceToHost)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = cudaErrorInvalidMemcpyDirection;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return cudaErrorInvalidMemcpyDirection;
		}
		dptr = (char*)dptr + offset;
		return cudaMemcpyAsync(dst, dptr, count, kind, stream);
	}

	__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) 
	{
		cudaError_t cudaErr = cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, 0);
		if(cudaErr != cudaSuccess)
			return cudaErr;
		return cudaStreamSynchronize(0);
	}

	__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) 
	{
		cudaError_t cudaErr = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, 0);
		if(cudaErr != cudaSuccess)
			return cudaErr;
		return cudaStreamSynchronize(0);
	}

	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		cudaError_t cudaErr = info.cudaLastError;
		GC_ThreadInfo[pid].cudaLastError = cudaSuccess;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaErr;
	}
	
	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		//printf("cudart device sync begin\n");
		CudaDeviceSynchronizeMsg_t msg(threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		//printf("cudart device sync end\n");
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaErr;
	}
	 
	__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSetLimit(cudaLimit limit, size_t value)
	{
		pthread_t pid= pthread_self();
		
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		
		cudaError_t cudaResu;
	  CudaDeviceSetLimitMsg_t msg(limit,value,threadId);
		mpi_error( MPI_Send(&msg, sizeof(CudaDeviceSetLimitMsg_t), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		MPI_Status status;
		mpi_error( MPI_Recv(&cudaResu, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, &status) );
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaResu;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaResu;
	}	
	//to be changed
	/*
	Explicitly destroys and cleans up all resources associated with the current device in the
	current process. Any subsequent API call to this device will reinitialize the device.
	Note that this function will reset the device immediately. It is the caller's responsibility to
	ensure that the device is not being accessed by any other host threads from the process
	when this function is called.
	*/
	__host__ cudaError_t CUDARTAPI cudaDeviceReset(void) 
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		cudaError_t err = cudaDeviceSynchronize();
		if(err != cudaSuccess)
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError = err;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			return err;
		}
		CudaDeviceResetMsg_t msg(threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, GC_DeviceList[device], NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, GC_DeviceList[device], threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
		return cudaSuccess;
	}
	__host__ cudaError_t cudaEventCreate (cudaEvent_t *event)
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		int dest = GC_DeviceList[info.currentDevice];
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		
		CudaEventCreateMsg_t msg(threadId);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		CudaEventCreateAckMsg_t amsg;
		mpi_error( MPI_Recv(&amsg, sizeof(amsg), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		if(amsg.status == cudaSuccess) {
			*event = amsg.event;
			pthread_mutex_lock(&GC_EventTagMutex);
			GC_EventTag[device][amsg.event] = EventAttr(device,NULL);
			pthread_mutex_unlock(&GC_EventTagMutex);
		}
		else
			*event = NULL;
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = amsg.status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		printf("cudaEventCreate finish event = %x, status = %d\n",*event,amsg.status);
		return amsg.status;
	}
	
	__host__ cudaError_t cudaEventDestroy (cudaEvent_t event)
	{
		printf("cudaEventDestroy event = %x\n",event);
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		int dest = GC_DeviceList[info.currentDevice];
		pthread_mutex_unlock(&GC_ThreadInfoMutex);

		pthread_mutex_lock(&GC_EventTagMutex);
		std::map<cudaEvent_t, EventAttr>::iterator it = GC_EventTag[device].find(event);
		if(it == GC_EventTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_EventTagMutex);
			return cudaErrorInvalidValue;
		}
		GC_EventTag[device].erase(event);
		pthread_mutex_unlock(&GC_EventTagMutex);
		
		CudaEventDestroyMsg_t msg(threadId,event);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		return cudaErr;
	}

	__host__ cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		int dest = GC_DeviceList[info.currentDevice];
		pthread_mutex_unlock(&GC_ThreadInfoMutex);

		pthread_mutex_lock(&GC_EventTagMutex);
		std::map<cudaEvent_t, EventAttr>::iterator it = GC_EventTag[device].find(event);
		if(it == GC_EventTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_EventTagMutex);
			return cudaErrorInvalidValue;
		}
		pthread_mutex_unlock(&GC_EventTagMutex);
		
		if(stream == NULL)
			stream = GC_StreamNull[device];
		pthread_mutex_lock(&GC_StreamTagMutex);
		std::map<cudaStream_t, StreamAttr>::iterator it1 = GC_StreamTag[device].find(stream);
		if(it1 == GC_StreamTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidResourceHandle;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_StreamTagMutex);
			return cudaErrorInvalidResourceHandle;
		}
		pthread_mutex_unlock(&GC_StreamTagMutex);

		printf("cudaEventRecord event = %x,stream = %x\n",event,stream);
		CudaEventRecordMsg_t msg(threadId,event,stream);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		printf("cudaEventRecord status = %d\n",cudaErr);
		return cudaErr;
	}

	__host__ cudaError_t cudaEventSynchronize(cudaEvent_t event)
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		int dest = GC_DeviceList[info.currentDevice];
		pthread_mutex_unlock(&GC_ThreadInfoMutex);

		pthread_mutex_lock(&GC_EventTagMutex);
		std::map<cudaEvent_t, EventAttr>::iterator it = GC_EventTag[device].find(event);
		if(it == GC_EventTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_EventTagMutex);
			return cudaErrorInvalidValue;
		}
		pthread_mutex_unlock(&GC_EventTagMutex);

		printf("cudaEventSynchronize event = %x\n",event);
		CudaEventSynchronizeMsg_t msg(threadId,event);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		cudaError_t cudaErr;
		mpi_error( MPI_Recv(&cudaErr, sizeof(cudaError_t), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = cudaErr;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		printf("cudaEventSynchronize status = %d\n",cudaErr);
		return cudaErr;
	}
	__host__ cudaError_t cudaEventElapsedTime (float *ms,cudaEvent_t start, cudaEvent_t end)
	{
		pthread_t pid= pthread_self();
		pthread_mutex_lock(&GC_ThreadInfoMutex)	;
		threadInfoTest(pid);
		ThreadInfo info = GC_ThreadInfo[pid];
		int threadId = info.threadId;
		int device = info.currentDevice;
		int dest = GC_DeviceList[info.currentDevice];
		pthread_mutex_unlock(&GC_ThreadInfoMutex);

		pthread_mutex_lock(&GC_EventTagMutex);
		std::map<cudaEvent_t, EventAttr>::iterator it = GC_EventTag[device].find(start);
		if(it == GC_EventTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_EventTagMutex);
			return cudaErrorInvalidValue;
		}
		std::map<cudaEvent_t, EventAttr>::iterator it1 = GC_EventTag[device].find(end);
		if(it1 == GC_EventTag[device].end())
		{
			pthread_mutex_lock(&GC_ThreadInfoMutex)	;
			GC_ThreadInfo[pid].cudaLastError  = cudaErrorInvalidValue;
			pthread_mutex_unlock(&GC_ThreadInfoMutex)	;
			
			pthread_mutex_unlock(&GC_EventTagMutex);
			return cudaErrorInvalidValue;
		}
		pthread_mutex_unlock(&GC_EventTagMutex);

		CudaEventElapsedTimeMsg_t msg(threadId,start,end);
		mpi_error( MPI_Send(&msg, sizeof(msg), MPI_BYTE, dest, NullTag, GC_DeviceComm) );
		CudaEventElapsedTimeAckMsg_t amsg;
		mpi_error( MPI_Recv(&amsg, sizeof(amsg), MPI_BYTE, dest, threadId << 16, GC_DeviceComm, MPI_STATUS_IGNORE) );
		*ms = amsg.ms;
		pthread_mutex_lock(&GC_ThreadInfoMutex);
		GC_ThreadInfo[pid].cudaLastError = amsg.status;
		pthread_mutex_unlock(&GC_ThreadInfoMutex);
		printf("cudaEventElapsedTime start = %x,end = %x,ms = %lf,status = %d\n",start,end,amsg.ms,amsg.status);
		return amsg.status;
	}
}

