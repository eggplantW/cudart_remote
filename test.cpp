
#include "cuda_runtime_api.h"
#include "ReduMax.h"
#include <iostream>
#include <cstdlib>
#include <float.h>
#include "RemoteAssistant.h"
#include <cstring>

using namespace std;

#define DATASIZE 100000

int main(int argc, char *argv[]){
	if(argc < 2)
		cout << "too few args" << endl;
	cout << "begin" << endl;
	if(strcmp(argv[1], "1") == 0) {
		RemoteAssistant remoteAssistant;
		remoteAssistant.Init(&argc, &argv);
		remoteAssistant.Run();
		remoteAssistant.Finalize();
	}
	else {
		cout << "main begin" << endl;
		cuda_remote_init(&argc, &argv);
		bool stuckPoint = false;
		while(stuckPoint);
		float *h_data, *d_data1, *d_data2;
		float resu[2];
		h_data = new float[DATASIZE];
		for(int i = 0; i < DATASIZE; i++)
			h_data[i] = rand();
		float max_resu = FLT_MIN;
		for(int i = 0; i < DATASIZE; i++) {
			if(h_data[i] > max_resu)
				max_resu = h_data[i];
		}
		cuda_error( cudaSetDevice(0) );
		cout << "CuSetDevice" << endl;
		cuda_error( cudaMalloc((void **)&d_data1, DATASIZE/2*sizeof(float)) );
		cout << "CudaMalloc" << endl;
		cuda_error( cudaMemcpy(d_data1, h_data, DATASIZE/2*sizeof(float), cudaMemcpyHostToDevice) );
		cout << "CudaMemcpy" << endl;
		resu[0] = ReduMaxFloat(DATASIZE/2, 0, d_data1);
		cout << "ReduMaxFloat" << endl;
		cuda_error( cudaFree(d_data1) );
		cout << "cudaFree" << endl;
		cuda_error( cudaSetDevice(1) );
		cout << "CuSetDevice" << endl;
		cuda_error( cudaMalloc((void **)&d_data2, DATASIZE/2*sizeof(float)) );
		cout << "CudaMalloc" << endl;
		cuda_error( cudaMemcpy(d_data2, h_data+DATASIZE/2, DATASIZE/2*sizeof(float), cudaMemcpyHostToDevice) );
		cout << "CudaMemcpy" << endl;
		resu[1] = ReduMaxFloat(DATASIZE/2, 0, d_data2);
		cout << "ReduMaxFloatR" << endl;
		cuda_error( cudaFree(d_data2) );

		std::cout << max_resu << std::endl;
		std::cout << ((resu[0] > resu[1])? resu[0] : resu[1]) << std::endl;
		cuda_remote_finalize();
	}

}
