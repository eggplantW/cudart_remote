#!/bin/sh
mpiexec -genv MV2_ENABLE_AFFINITY 0 -genv MV2_USE_RDMA_FAST_PATH 0 -genv MV2_SUPPORT_DPM 1 -genv GC_DEVICE_HOSTS gpuhf -genv GC_DEVICE_LIST "0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 " -f hf -n 1 -env CUDA_VISIBLE_DEVICES "" ./vectorAdd
