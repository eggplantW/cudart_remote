
#include "cuda_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "Functions.h"

#define SPHBSIZE 256
#define DG_ReduMaxFloat

void CudaSafeCall(cudaError_t err){
	if(err != cudaSuccess){
		printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

//==============================================================================
/// Returns dimensions of gridsize according to parameters.
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize;//-Total number of blocks to be launched.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//==============================================================================
/// Reduction using maximum of float values in shared memory for a warp.
//==============================================================================
__device__ __forceinline__ void KerReduMaxFloatWarp(volatile float* sdat,unsigned tid) {
  if(blockDim.x>=64)sdat[tid]=max(sdat[tid],sdat[tid+32]);
  if(blockDim.x>=32)sdat[tid]=max(sdat[tid],sdat[tid+16]);
  if(blockDim.x>=16)sdat[tid]=max(sdat[tid],sdat[tid+8]);
  if(blockDim.x>=8)sdat[tid]=max(sdat[tid],sdat[tid+4]);
  if(blockDim.x>=4)sdat[tid]=max(sdat[tid],sdat[tid+2]);
  if(blockDim.x>=2)sdat[tid]=max(sdat[tid],sdat[tid+1]);
}

//==============================================================================
/// Accumulates the summation of n values of array dat[], storing the result in the beginning of res[].
/// As many positions of res[] as blocks are used, storing the final result in res[0]).
//==============================================================================
extern "C" __global__ void KerReduMaxFloat(unsigned n,unsigned ini,const float *dat,float *res){
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  //unsigned c=blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini]: -FLT_MAX);
  __syncthreads();
  for(int i = blockDim.x >> 1; i > 32; i = i >> 1){
	  if(tid < i)
		  sdat[tid] = sdat[tid] > sdat[tid+i]? sdat[tid] : sdat[tid+i];
	  __syncthreads();
  }
  if(tid<32)KerReduMaxFloatWarp(sdat,tid);
  if(tid==0)res[blockIdx.y*gridDim.x + blockIdx.x]=sdat[0];
}

void C_KerReduMaxFloat(unsigned n, unsigned ini, const float *dat, float *res, dim3 Dg, dim3 Db, size_t Ns = 0, cudaStream_t S = NULL){
	KerReduMaxFloat<<<Dg, Db, Ns, S>>>(n, ini, dat, res);
}

//==============================================================================
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] msut be >= N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
//==============================================================================
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data){
  //printf("[ReduMaxF ndata:%d  SPHBSIZE:%d]\n",ndata,SPHBSIZE);
  unsigned n=ndata,ini=inidata;
  unsigned smemSize=SPHBSIZE*sizeof(float);
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  //printf("n:%d  n_blocks:%d]\n",n,n_blocks);
  float *dat=data;
  float *resu;
  CudaSafeCall(cudaMalloc(&resu, sizeof(float)*(n_blocks+(n_blocks+SPHBSIZE-1)/SPHBSIZE)));
  float *res = resu;
  float *res1 = resu;
  float *res2 = resu+n_blocks;
  bool res_flag = true;
  while(true){
    //printf("##>ReduMaxF n:%d  n_blocks:%d  ini:%d\n",n,n_blocks,ini);
    //printf("##>ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
	CudaSafeCall(cudaGetLastError());
    KerReduMaxFloat<<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
    cudaDeviceSynchronize();
    CudaSafeCall(cudaGetLastError());
    //KerReduMaxF<SPHBSIZE><<<n_blocks,SPHBSIZE,smemSize>>>(n,dat,res);
    //CheckErrorCuda("#>ReduMaxF KerReduMaxF  failed.");
    if(n_blocks < 2)
    	break;
    n=n_blocks; ini=0;
    sgrid=GetGridSize(n,SPHBSIZE);
    n_blocks=sgrid.x*sgrid.y;
    dat = res;
    if(res_flag)
    	res = res2;
    else
    	res = res1;
    res_flag = !res_flag;
  }
  float resf;
  if(ndata>1)CudaSafeCall(cudaMemcpy(&resf,res,sizeof(float),cudaMemcpyDeviceToHost));
  else CudaSafeCall(cudaMemcpy(&resf,data,sizeof(float),cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(resu));
  //CheckErrorCuda("#>ReduMaxF cudaMemcpy  failed.");
#ifdef DG_ReduMaxFloat
  if(1){ //-Checks the reduction <DEBUG>
    float *vdat=new float[ndata];
    cudaMemcpy(vdat,data+inidata,sizeof(float)*ndata,cudaMemcpyDeviceToHost);
    float maxi=vdat[0];
    //for(unsigned c=0;c<ndata;c++){ printf("ReduMaxF>vdat[%u]=%f\n",c,vdat[c]); }
    for(unsigned c=1;c<ndata;c++)if(maxi<vdat[c])maxi=vdat[c];
    if(resf!=maxi){
      printf("ReduMaxF>ERRORRRRR... Maximo:; %f; %f\n",resf,maxi);
      printf("ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
      exit(0);
    }
    delete[] vdat;
  }
#endif
  return(resf);
}

