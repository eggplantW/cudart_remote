/*
 * Types.h
 *
 *  Created on: 2015-6-19
 *      Author: makai
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

enum MesgTag { NullTag = 0, DeviceInitTag,
				CudaModuleLoadTag, CudaModuleLoadAckTag,
				CudaModuleGetFuncTag, CudaModuleGetFuncAckTag,
				CudaModuleGetSymbolTag, CudaModuleGetSymbolAckTag,
				CudaSetDeviceTag, CudaSetDeviceAckTag,
				CudaGetDevicePropertiesTag, CudaGetDevicePropertiesAckTag,
				CudaDeviceSetCacheConfigTag, CudaDeviceSetCacheConfigAckTag,
				CudaGetErrorStringTag, CudaGetErrorStringAckTag,
				CudaStreamCreateTag, CudaStreamCreateAckTag,
				CudaStreamDestroyTag, CudaStreamDestroyAckTag,
				CudaStreamSynchronizeTag, CudaStreamSynchronizeAckTag,
				CudaMallocTag, CudaMallocAckTag,
				CudaMemsetTag,
				CudaFreeTag, CudaFreeAckTag,
				CudaMemcpyTag, CudaMemcpyAckTag,
				CudaMemcpyAsyncTag, CudaMemcpyAsyncAckTag,
				CudaMemcpyAsyncData,
				CudaConfigureCallTag, CudaConfigureCallAckTag,
				CudaSetupArgumentTag, CudaSetupArgumentAckTag,
				CudaLaunchTag, CudaLaunchAckTag,
				CudaLaunchKernelTag, CudaLaunchKernelAckTag,
				CudaGetLastErrorTag, CudaGetLastErrorAckTag,
				CudaDeviceSynchronizeTag, CudaDeviceSynchronizeAckTag,
				CudaModuleCompleteTag, DataTag, ParameterTag,
				CudaDeviceResetTag,CudaDeviceSetLimitTag,
				CudaEventCreateTag,CudaEventDestroyTag,CudaEventRecordTag,
				CudaEventSynchronizeTag,CudaEventElapsedTimeTag,
				ThreadCallTag };

enum GC_StreamFlag { NullStreamFlag, DefaultStreamFlag, NonblockingStreamFlag };

struct Msg_t {
	MesgTag msgTag;
	Msg_t(MesgTag _msgTag) : msgTag(_msgTag) { }
};

struct CudaModuleGetFuncMsg_t : Msg_t {
	CUmodule module;
	int threadId;
	CudaModuleGetFuncMsg_t(CUmodule _module,int _threadId) : Msg_t(CudaModuleGetFuncTag), module(_module), threadId(_threadId) { }
};

struct CudaModuleGetSymbolMsg_t : Msg_t {
	CUmodule module;
	int threadId;
	CudaModuleGetSymbolMsg_t(CUmodule _module,int _threadId) : Msg_t(CudaModuleGetSymbolTag), module(_module), threadId(_threadId) { }
};

struct CudaGetDevicePropertiesMsg_t : Msg_t {
	int threadId;
	CudaGetDevicePropertiesMsg_t(int _threadId) : Msg_t(CudaGetDevicePropertiesTag), threadId(_threadId) { }
};

struct CudaGetDevicePropertiesAckMsg_t {
	cudaError_t cudaStat;
	struct cudaDeviceProp prop;
};

struct CudaDeviceSetCacheConfigMsg_t : Msg_t {
	cudaFuncCache cacheConfig;
	int threadId;
	CudaDeviceSetCacheConfigMsg_t(cudaFuncCache _cacheConfig,int _threadId) : Msg_t(CudaDeviceSetCacheConfigTag), cacheConfig(_cacheConfig),threadId(_threadId) { }
};

struct CudaGetErrorStringMsg_t : Msg_t{
	cudaError_t err;
	int threadId;
	CudaGetErrorStringMsg_t(cudaError_t _err,int _threadId) : Msg_t(CudaGetErrorStringTag), err(_err),threadId(_threadId) { }
};

struct CudaStreamCreateMsg_t : Msg_t {
	enum GC_StreamFlag flag;
	int threadId;
	CudaStreamCreateMsg_t(enum GC_StreamFlag _flag,int _threadId) : Msg_t(CudaStreamCreateTag), flag(_flag), threadId(_threadId) { }
	CudaStreamCreateMsg_t() : Msg_t(CudaStreamCreateTag), flag(DefaultStreamFlag) { }
};

struct CudaStreamCreateAckMsg_t {
	cudaStream_t stream;
	int streamTag;
	cudaError_t status;
};

struct CudaStreamDestroyMsg_t : Msg_t{
	cudaStream_t stream;
	int threadId;
	CudaStreamDestroyMsg_t(cudaStream_t _stream,int _threadId) : Msg_t(CudaStreamDestroyTag), stream(_stream), threadId(_threadId) { }
};

struct CudaStreamSynchronizeMsg_t : Msg_t {
	cudaStream_t stream;
	int threadId;
	CudaStreamSynchronizeMsg_t(cudaStream_t _stream,int _threadId) : Msg_t(CudaStreamSynchronizeTag), stream(_stream), threadId(_threadId) { }
};

struct CudaMallocMsg_t : public Msg_t {
	size_t size;
	int threadId;
	CudaMallocMsg_t(size_t _size, int _threadId) :
		Msg_t(CudaMallocTag),
		size(_size), threadId(_threadId) { }
};

struct CudaMemsetMsg_t : public Msg_t{
	void *dptr;
	int value;
	size_t size;
	int threadId;
	CudaMemsetMsg_t(void *_dptr, int _value, int _size, int _threadId):
		Msg_t(CudaMemsetTag),dptr(_dptr),value(_value),size(_size),threadId(_threadId){}
};

struct CudaFreeMsg_t : public Msg_t {
	void *dptr;
	int threadId;
	CudaFreeMsg_t(void *_dptr, int _threadId):
		Msg_t(CudaFreeTag),
		dptr(_dptr), threadId(_threadId) { }
};

struct CudaMemcpyMsg_t : public Msg_t {
	void *ptr;
	size_t count;
	cudaMemcpyKind kind;
	cudaStream_t stream;
	CudaMemcpyMsg_t(void *_ptr, size_t _count, enum cudaMemcpyKind _kind, cudaStream_t _stream) :
		Msg_t(CudaMemcpyTag), ptr(_ptr),
		count(_count), kind(_kind), stream(_stream) { }
};

struct CudaMemcpyAsyncMsg_t : public Msg_t {
	void *ptr;
	size_t count;
	cudaMemcpyKind kind;
	cudaStream_t stream;
	int threadId;
	CudaMemcpyAsyncMsg_t() : Msg_t(CudaMemcpyAsyncTag), ptr(NULL), count(0), kind(cudaMemcpyHostToDevice), stream(NULL), threadId(-1) { }
	CudaMemcpyAsyncMsg_t(void *_ptr, size_t _count, enum cudaMemcpyKind _kind, cudaStream_t _stream, int _threadId) : Msg_t(CudaMemcpyAsyncTag), ptr(_ptr), count(_count), kind(_kind), stream(_stream), threadId(_threadId) { }
};

struct CudaMallocAckMsg_t {
	void *dptr;
	cudaError_t status;
};

struct CudaConfigureCallMsg_t : public Msg_t {
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
	int threadId;
	CudaConfigureCallMsg_t(dim3 _gridDim, dim3 _blockDim, size_t _sharedMem, cudaStream_t _stream, int _threadId) :
		Msg_t(CudaConfigureCallTag),
		gridDim(_gridDim), blockDim(_blockDim),
		sharedMem(_sharedMem), stream(_stream), threadId(_threadId) { }
};

struct CudaSetupArgumentMsg_t : public Msg_t {
	size_t size;
	size_t offset;
	int threadId;
	CudaSetupArgumentMsg_t(size_t _size, size_t _offset, int _threadId):
		Msg_t(CudaSetupArgumentTag),
		size(_size), offset(_offset), threadId(_threadId) { }
};


struct CudaModuleLoadMsg_t: public Msg_t{
	int threadId;
	CudaModuleLoadMsg_t(int _threadId):Msg_t(CudaModuleLoadTag),threadId(_threadId){}
};

struct CudaLaunchMsg_t: public Msg_t {
	CUmodule module;
	CUfunction hfunc;
	int threadId;
	CudaLaunchMsg_t(CUmodule _module, CUfunction _hfunc, int _threadId): Msg_t(CudaLaunchTag), module(_module), hfunc(_hfunc), threadId(_threadId) { }
};

struct CudaDeviceResetMsg_t:public Msg_t{
	int threadId;
	CudaDeviceResetMsg_t(int _threadId):Msg_t(CudaDeviceResetTag),threadId(_threadId){}
};

struct CudaDeviceSynchronizeMsg_t:public Msg_t{
	int threadId;
	CudaDeviceSynchronizeMsg_t(int _threadId):Msg_t(CudaDeviceSynchronizeTag),threadId(_threadId){}
};

struct CudaDeviceSetLimitMsg_t:public Msg_t{
	cudaLimit limit;
	size_t value;
	int threadId;
	CudaDeviceSetLimitMsg_t(cudaLimit _limit,size_t _value,int _threadId):Msg_t(CudaDeviceSetLimitTag),limit(_limit),value(_value),threadId(_threadId){}
};

struct CudaEventCreateMsg_t : Msg_t {
	int threadId;
	CudaEventCreateMsg_t(int _threadId) : Msg_t(CudaEventCreateTag), threadId(_threadId) { }
	CudaEventCreateMsg_t() : Msg_t(CudaEventCreateTag) { }
};

struct CudaEventCreateAckMsg_t {
	cudaEvent_t event;
	cudaError_t status;
};
struct CudaEventDestroyMsg_t : Msg_t {
	int threadId;
	cudaEvent_t event;
	CudaEventDestroyMsg_t(int _threadId,cudaEvent_t _event) : Msg_t(CudaEventDestroyTag), threadId(_threadId),event(_event) { }
	CudaEventDestroyMsg_t() : Msg_t(CudaEventDestroyTag) { }
};
struct CudaEventRecordMsg_t : Msg_t {
	int threadId;
	cudaEvent_t event;
	cudaStream_t stream;
	CudaEventRecordMsg_t(int _threadId,cudaEvent_t _event,cudaStream_t _stream) : Msg_t(CudaEventRecordTag), threadId(_threadId),event(_event),stream(_stream) { }
	CudaEventRecordMsg_t() : Msg_t(CudaEventRecordTag) { }
};
struct CudaEventSynchronizeMsg_t : Msg_t {
	int threadId;
	cudaEvent_t event;
	CudaEventSynchronizeMsg_t(int _threadId,cudaEvent_t _event) : Msg_t(CudaEventSynchronizeTag), threadId(_threadId),event(_event){ }
	CudaEventSynchronizeMsg_t() : Msg_t(CudaEventSynchronizeTag) { }
};
struct CudaEventElapsedTimeMsg_t : Msg_t {
	int threadId;
	cudaEvent_t start;
	cudaEvent_t end;
	CudaEventElapsedTimeMsg_t(int _threadId,cudaEvent_t _start,cudaEvent_t _end) : Msg_t(CudaEventElapsedTimeTag), threadId(_threadId),start(_start),end(_end){ }
	CudaEventElapsedTimeMsg_t() : Msg_t(CudaEventElapsedTimeTag) { }
};
struct CudaEventElapsedTimeAckMsg_t{
	float ms;
	cudaError_t status;
};



#endif /* TYPES_H_ */
