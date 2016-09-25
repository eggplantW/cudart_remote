/*
 * Defines.h
 *
 *  Created on: 2015-6-18
 *      Author: makai
 */

#ifndef DEFINES_H_
#define DEFINES_H_

//#define _NO_PIPELINE

#define BufferSizeInter 1024
#define BufferSize 1024
#define ErrorStringLength 256
#define MAX_STREAM_NUM_PER_THREAD 10

typedef void *(*thread_func_t)(void *);

#define HOST_BUFFER_INIT_SIZE (1<<21)

#endif /* DEFINES_H_ */
