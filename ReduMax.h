/*
 * ReduMax.h
 *
 *  Created on: 2015-6-12
 *      Author: makai
 */

#ifndef REDUMAX_H_
#define REDUMAX_H_

void CudaSafeCall(cudaError_t err);

template<unsigned blockSize>
void C_KerReduMaxFloat(unsigned n, unsigned ini, const float *dat, float *res, dim3 Dg, dim3 Db, size_t Ns = 0, cudaStream_t S = NULL);

float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data);


#endif /* REDUMAX_H_ */
