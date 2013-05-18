#ifndef __CLSTRUCT_H__
#define __CLSTRUCT_H__

#include <cstdlib>
#include <CL/cl.h>
struct CLStruct {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue cqueue;
	cl_program program;
	cl_kernel gaussian, diff;
};

#define ABORT_IF(COND, STR)\
	if (COND) {\
		printf(STR);\
		abort();\
	}

#endif /* end of include guard: __CLSTRUCT_H__ */
