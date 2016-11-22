/*
 * 2016.11.11. - 11.13. TEST OK
 * Allocation of array of pointers
 * Allocation of each element in the array
 */
#if 0
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand, malloc       */
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "type.h"
#include "macro.h"
#include "redutil2.h"

using namespace redutil2;

namespace kernel_test
{
__global__
	void print_array(int **a, uint32_t n_vct)
	{
		const int tid = threadIdx.x;

		if (0 == tid)
		{
			for (uint32_t i = 0; i < n_vct; i++)
			{
				printf("[%u]: %p *(_+%u): %p\n", i, a[i], i, *(a+i));
			}
		}
	}

__global__
	void print_array(int *a, uint32_t n_arr)
	{
		const int tid = threadIdx.x;

		if (0 == tid)
		{
			for (uint32_t i = 0; i < n_arr; i++)
			{
				printf("\t[%u]: %d\n", i, a[i]);
			}
		}
	}
} /* kernel_test */

void print_array(int **a, uint32_t n_vct)
{
	for (uint32_t i = 0; i < n_vct; i++)
	{
		printf(" +%u: %p\t", i, a+i);
		printf("[%u]: %p *( +%u): %p\n", i, a[i], i, *(a+i));
	}
}

void print_array(int *a, uint32_t n_arr)
{
	for (uint32_t i = 0; i < n_arr; i++)
	{
		printf("\t[%u]: %d\n", i, a[i]);
	}
}

int main()
{
	static const uint32_t n_vct = 5;
	static const uint32_t n_arr = 9;

	int** h_k = NULL;
	int** d_k = NULL;
	int** tmp = NULL;

	try
	{
		printf("h_k: %p\t", h_k);

		// Allocate HOST memory
		ALLOCATE_HOST_VECTOR((void**)&h_k, n_vct*sizeof(int*));
		printf("after allocation: %p\n", h_k);

		for (uint32_t i = 0; i < n_vct; i++)
		{
			printf("h_k[%u]: %p\t", i, h_k[i]);
			ALLOCATE_HOST_VECTOR((void**)(h_k + i), n_arr*sizeof(int));
			printf("after allocation: %p\n", h_k[i]);
			print_array(*(h_k + i), n_arr);
		}

		printf("tmp: %p\t", tmp);
		ALLOCATE_HOST_VECTOR((void**)&tmp, n_vct*sizeof(int*));
		printf("after allocation: %p\n", tmp);

		// Allocate DEVICE memory
		printf("d_k: %p\t", d_k);
		ALLOCATE_DEVICE_VECTOR((void**)(&d_k), n_vct*sizeof(int*));
		printf("after allocation: %p\n", d_k);

		for (uint32_t i = 0; i < n_vct; i++)
		{
			printf("tmp[%u]: %p\t", i, tmp[i]);
			ALLOCATE_DEVICE_VECTOR((void**)(tmp + i), n_arr*sizeof(int));
			printf("after allocation: %p\n", tmp[i]);
			kernel_test::print_array<<<1,  1>>>(*(tmp + i), n_arr);
			cudaThreadSynchronize();
		}
		CUDA_SAFE_CALL(cudaMemcpy(d_k, tmp, n_vct * sizeof(int*), cudaMemcpyHostToDevice));
		kernel_test::print_array<<<1,  1>>>(d_k, n_vct);
		cudaThreadSynchronize();


		// Populate data
		for (uint32_t i = 0; i < n_vct; i++)
		{
			for (uint32_t j = 0; j < n_arr; j++)
			{
				*(*(h_k+i)+j) = i*10 + j;
			}
			printf("h_k[%u]: %p\n", i, h_k[i]);
			print_array(*(h_k + i), n_arr);
			printf("\n");

			printf("tmp[%u]: %p\n", i, tmp[i]);
			CUDA_SAFE_CALL(cudaMemcpy(tmp[i], h_k[i], n_arr * sizeof(int), cudaMemcpyHostToDevice));
			kernel_test::print_array<<<1,  1>>>(tmp[i], n_arr);
			cudaThreadSynchronize();
		}

		// Deallocate memory
		for (uint32_t i = 0; i < n_vct; i++)
		{
			FREE_HOST_VECTOR((void**)(h_k + i));
			FREE_DEVICE_VECTOR((void**)(tmp + i));
		}
		FREE_HOST_VECTOR((void**)&h_k);
		FREE_HOST_VECTOR((void**)&tmp);
		FREE_DEVICE_VECTOR((void**)&d_k);
	}
	catch (const std::string& msg)
	{
		std::cerr << "Error: " << msg << std::endl;
	}

	return 0;
}
#endif


/*
 * 2016.11.13. - 11.13.  TEST OK
 * Compute the linear combination of arrays on the DEVICE
 * and comapre the results those computed on the HOST
 */
#if 1
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand, malloc       */
#include <time.h>       /* time                      */
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "type.h"
#include "macro.h"
#include "redutil2.h"

using namespace redutil2;

namespace kernel_test
{
__global__
void print_array(var_t *a, uint32_t n_arr)
{
	const int tid = threadIdx.x;

	if (0 == tid)
	{
		for (uint32_t i = 0; i < n_arr; i++)
		{
			printf("\t[%u]: %g\n", i, a[i]);
		}
	}
}

//! Calculate the special case of linear combination of vectors, a[i] = b[i] + sum (coeff[j] * c[j][i])
/*
	\param a     vector which will contain the result
	\param b     vector to which the linear combination will be added
	\param c     vectors which will linear combined
	\param coeff vector which contains the weights (coefficients)
	\param n_vct the number of vectors to combine
	\param n_var the number of elements in the vectors
*/
__global__
void calc_lin_comb_s(var_t* a, const var_t* b, const var_t* const *c, const var_t* coeff, uint16_t n_vct, uint32_t n_var)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n_var)
	{
		var_t d = 0.0;
		for (uint16_t j = 0; j < n_vct; j++)
		{
			if (0.0 == coeff[j])
			{
				continue;
			}
			d += coeff[j] * c[j][tid];
		}
		a[tid] = b[tid] + d;
	}
}
} /* kernel_test */

int main()
{
	static const uint32_t n_vct = 5;
	static const uint32_t n_arr = 3000;

	var_t** h_k = NULL;
	var_t** d_k = NULL;
	var_t** tmp = NULL;

	var_t* h_a = NULL;
	var_t* h_a0 = NULL;     // Will hold a copy of d_a
	var_t* h_b = NULL;
	var_t* h_coeff = NULL;

	var_t* d_a = NULL;
	var_t* d_b = NULL;
	var_t* d_coeff = NULL;

	try
	{
		// Allocate HOST memory
		ALLOCATE_HOST_VECTOR((void**)&h_k, n_vct*sizeof(var_t*));
		for (uint32_t i = 0; i < n_vct; i++)
		{
			ALLOCATE_HOST_VECTOR((void**)(h_k + i), n_arr*sizeof(var_t));
		}
		ALLOCATE_HOST_VECTOR((void**)&tmp, n_vct*sizeof(var_t*));

		ALLOCATE_HOST_VECTOR((void**)&h_a,     n_arr*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_a0,    n_arr*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_b,     n_arr*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_coeff, n_vct*sizeof(var_t));

		// Allocate DEVICE memory
		ALLOCATE_DEVICE_VECTOR((void**)(&d_k), n_vct*sizeof(var_t*));
		for (uint32_t i = 0; i < n_vct; i++)
		{
			ALLOCATE_DEVICE_VECTOR((void**)(tmp + i), n_arr*sizeof(var_t));
		}
		CUDA_SAFE_CALL(cudaMemcpy(d_k, tmp, n_vct * sizeof(var_t*), cudaMemcpyHostToDevice));

		ALLOCATE_DEVICE_VECTOR((void**)&d_a,     n_arr*sizeof(var_t));
		ALLOCATE_DEVICE_VECTOR((void**)&d_b,     n_arr*sizeof(var_t));
		ALLOCATE_DEVICE_VECTOR((void**)&d_coeff, n_vct*sizeof(var_t));

		// Populate data
		srand(time(NULL));
		for (uint32_t i = 0; i < n_vct; i++)
		{
			for (uint32_t j = 0; j < n_arr; j++)
			{
				var_t r = (var_t)rand()/RAND_MAX;    //returns a pseudo-random integer between 0 and RAND_MAX			
				*(*(h_k+i)+j) = r;
			}
			CUDA_SAFE_CALL(cudaMemcpy(tmp[i], h_k[i], n_arr * sizeof(var_t), cudaMemcpyHostToDevice));
		}
		for (uint32_t j = 0; j < n_arr; j++)
		{
			h_a[j] = 0;
			h_b[j] = 0;
		}
		for (uint32_t j = 0; j < n_vct; j++)
		{
			h_coeff[j] = 1;
		}
		h_coeff[4] = -1;

		CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, n_arr * sizeof(var_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, n_arr * sizeof(var_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_coeff, h_coeff, n_vct * sizeof(var_t), cudaMemcpyHostToDevice));

		// Test the tools::calc_lin_comb_s() and gpu_calc_lin_comb_s() functions
		// Compute a[i] = b[i] + f*c[i]
		{
			printf("Compute a[i] = b[i] + f*c[i]\n\n");
			var_t f = 2.0;
			var_t *h_c = *h_k;
			tools::calc_lin_comb_s(h_a, h_b, h_c, f, n_arr);

			var_t *d_c = *tmp;
			gpu_calc_lin_comb_s(   d_a, d_b, d_c, f, n_arr, 0, false);

			//printf("h_a:\n");
			//print_array("", n_arr, h_a, PROC_UNIT_CPU);

			//printf("d_a:\n");
			//print_array("", n_arr, d_a, PROC_UNIT_GPU);

			CUDA_SAFE_CALL(cudaMemcpy(h_a0, d_a, n_arr * sizeof(var_t), cudaMemcpyDeviceToHost));

			for (uint32_t j = 0; j < n_arr; j++)
			{
				if (0 != fabs(h_a[j] - h_a0[j]))
				{
					printf("Difference: j = %6u : %g\n", j, h_a[j] - h_a0[j]);
				}
			}
		}

		// Test the tools::calc_lin_comb_s() and gpu_calc_lin_comb_s() functions
		// Compute a[i] = b[i] + sum (coeff[j] * c[j][i])
		{
			printf("Compute a[i] = b[i] + sum (coeff[j] * c[j][i])\n\n");
			tools::calc_lin_comb_s(h_a, h_b, h_k, h_coeff, n_vct, n_arr);
			gpu_calc_lin_comb_s(   d_a, d_b, d_k, d_coeff, n_vct, n_arr, 0, false);
	
			CUDA_SAFE_CALL(cudaMemcpy(h_a0, d_a, n_arr * sizeof(var_t), cudaMemcpyDeviceToHost));

			for (uint32_t j = 0; j < n_arr; j++)
			{
				if (0 != fabs(h_a[j] - h_a0[j]))
				{
					printf("Difference: j = %6u : %g\n", j, h_a[j] - h_a0[j]);
				}
			}
		}

		// Deallocate memory
		for (uint32_t i = 0; i < n_vct; i++)
		{
			FREE_HOST_VECTOR((void**)(h_k + i));
			FREE_DEVICE_VECTOR((void**)(tmp + i));
		}
		FREE_HOST_VECTOR((void**)&h_k);
		FREE_HOST_VECTOR((void**)&tmp);
		FREE_DEVICE_VECTOR((void**)&d_k);

		FREE_HOST_VECTOR((void**)&h_a);
		FREE_HOST_VECTOR((void**)&h_a0);
		FREE_HOST_VECTOR((void**)&h_b);
		FREE_HOST_VECTOR((void**)&h_coeff);

		FREE_DEVICE_VECTOR((void**)&d_a);
		FREE_DEVICE_VECTOR((void**)&d_b);
		FREE_DEVICE_VECTOR((void**)&d_coeff);
	}
	catch (const std::string& msg)
	{
		std::cerr << "Error: " << msg << std::endl;
	}

	std::cout << "Compute the linear combination of arrays on the DEVICE and comapre the results those computed on the HOST done.\n";

	return 0;
}
#endif


/*
 * 2016.11.14. - 
 * Gravitational interaction computations
 */
#if 0
/*
Premature optimization is the ROOT OF ALL EVIL. Always remember the three rules of optimization!

1. Don't optimize.
2. If you are an expert, see rule #1
3. If you are an expert and can justify the need, then use the following procedure:
 - Code it unoptimized
 - determine how fast is "Fast enough"--Note which user requirement/story requires that metric.
 - Write a speed test
 - Test existing code--If it's fast enough, you're done.
 - Recode it optimized
 - Test optimized code. IF it doesn't meet the metric, throw it away and keep the original.
 - If it meets the test, keep the original code in as comments
*/

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand, malloc       */
#include <time.h>       /* time                      */
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "type.h"
#include "macro.h"
#include "redutil2.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

using namespace redutil2;

// Global variables
uint32_t n_tpb = 128;

uint32_t n_obj = 0;
var_t* h_p = NULL;
var_t* d_p = NULL;

dim3 grid;
dim3 block;


namespace nbody_kernel
{
__global__
void calc_gravity_accel_naive(uint32_t n_obj, const var3_t* r, const nbp_t::param_t* p, var3_t* a)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (n_obj > i)
	{
		var3_t r_ij = {0, 0, 0};
		for (uint32_t j = 0; j < n_obj; j++)
		{
			if (i == j)
			{
				continue;
			}
			r_ij.x = r[j].x - r[i].x;
			r_ij.y = r[j].y - r[i].y;
			r_ij.z = r[j].z - r[i].z;

			var_t d2 = SQR(r_ij.x) + SQR(r_ij.y) + SQR(r_ij.z);
			var_t d = sqrt(d2);
			var_t d_3 = 1.0 / (d*d2);

			var_t s = p[j].mass * d_3;
			a[i].x += s * r_ij.x;
			a[i].y += s * r_ij.y;
			a[i].z += s * r_ij.z;
		}
		a[i].x *= K2;
		a[i].y *= K2;
		a[i].z *= K2;
	}
}

__global__
void calc_gravity_accel_naive_sym(uint32_t n_obj, const var3_t* r, const nbp_t::param_t* p, var3_t* a)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (n_obj > i)
	{
		var3_t r_ij = {0, 0, 0};
		for (uint32_t j = i+1; j < n_obj; j++)
		{
			r_ij.x = r[j].x - r[i].x;
			r_ij.y = r[j].y - r[i].y;
			r_ij.z = r[j].z - r[i].z;

			var_t d2 = SQR(r_ij.x) + SQR(r_ij.y) + SQR(r_ij.z);
			var_t d = sqrt(d2);
			var_t d_3 = 1.0 / (d*d2);

			var_t s = p[j].mass * d_3;
			a[i].x += s * r_ij.x;
			a[i].y += s * r_ij.y;
			a[i].z += s * r_ij.z;

			s = p[i].mass * d_3;
			a[j].x -= s * r_ij.x;
			a[j].y -= s * r_ij.y;
			a[j].z -= s * r_ij.z;
		}
		a[i].x *= K2;
		a[i].y *= K2;
		a[i].z *= K2;
	}
}

inline __host__ __device__
	var3_t body_body_interaction(var3_t riVec, var3_t rjVec, var_t mj, var3_t aiVec)
{
	var3_t dVec = {0.0, 0.0, 0.0};

	// compute d = r_i - r_j [3 FLOPS] [6 read, 3 write]
	dVec.x = rjVec.x - riVec.x;
	dVec.y = rjVec.y - riVec.y;
	dVec.z = rjVec.z - riVec.z;

	// compute norm square of d vector [5 FLOPS] [3 read, 1 write]
	var_t r2 = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);
	// compute norm of d vector [1 FLOPS] [1 read, 1 write] TODO: how long does it take to compute sqrt ???
	var_t r = sqrt(r2);
	// compute m_j / d^3 []
	var_t s = mj * 1.0 / (r2 * r);

	aiVec.x += s * dVec.x;
	aiVec.y += s * dVec.y;
	aiVec.z += s * dVec.z;

	return aiVec;
}

__global__
	void calc_gravity_accel_tile(interaction_bound int_bound, int tile_size, const var3_t* r, const nbp_t::param_t* p, var3_t* a)
{
	extern __shared__ var3_t sh_pos[];

	var3_t my_pos = {0.0, 0.0, 0.0};
	var3_t acc    = {0.0, 0.0, 0.0};

	// i is the index of the SINK body
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	// To avoid overruning the r buffer
	if (int_bound.sink.y > i)
	{
		my_pos = r[i];
	}
	for (int tile = 0; (tile * tile_size) < int_bound.source.y; tile++)
	{
		// src_idx is the index of the SOURCE body in the tile
		int src_idx = int_bound.source.x + tile * tile_size + threadIdx.x;
		// To avoid overruning the r buffer
		if (int_bound.source.y > src_idx)
		{
			sh_pos[threadIdx.x] = r[src_idx];
		}
		__syncthreads();
		// j is the index of the SOURCE body in the current tile
		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (int_bound.source.y <= int_bound.source.x + (tile * tile_size) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (i != int_bound.source.x + (tile * tile_size)+j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], p[src_idx].mass, acc);
			}
		}
		__syncthreads();
	}

	// To avoid overruning the a buffer
	if (int_bound.sink.y > i)
	{
		a[i] = acc;
	}
}
} /* nbody_kernel */


/* 
 *  -- Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both --
 * Returns the amount of microseconds elapsed since the UNIX epoch. Works on both
 * windows and linux.
 */
uint64_t GetTimeMs64()
{
#ifdef _WIN32
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	* to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64_t ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	//ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */
	ret /= 10; /* From 100 nano seconds (10^-7) to 1 microsecond (10^-6) intervals */

	return ret;
#else
	/* Linux */
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	//ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	//ret += (tv.tv_sec * 1000);
	/* Adds the seconds (10^0) after converting them to microseconds (10^-6) */
	ret += (tv.tv_sec * 1000000);

	return ret;
#endif
}


float gpu_calc_dy(uint32_t n_var, uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy, bool use_symm_prop)
{
	set_kernel_launch_param(n_var, n_tpb, grid, block);
		
	printf(" grid: (%4u, %4u, %4u)\n", grid.x, grid.y, grid.z);
	printf("block: (%4u, %4u, %4u)\n", block.x, block.y, block.z);

	var3_t* r = (var3_t*)y_temp;
	var3_t* a = (var3_t*)(dy + 3*n_obj);
	nbp_t::param_t* p = (nbp_t::param_t*)d_p;

	cudaEvent_t t0, t1;
	CUDA_SAFE_CALL(cudaEventCreate(&t0));
	CUDA_SAFE_CALL(cudaEventCreate(&t1));

	CUDA_SAFE_CALL(cudaEventRecord(t0));
	// Clear the acceleration array: the += op can be used
	CUDA_SAFE_CALL(cudaMemset(a, 0, n_obj*sizeof(var3_t)));

	// Copy the velocities into dy
	// TODO: implement the asynchronous version of cudaMemcpy: Performace ??
	CUDA_SAFE_CALL(cudaMemcpy(dy, y_temp + 3*n_obj, 3*n_obj*sizeof(var_t), cudaMemcpyDeviceToDevice));

	if (false == use_symm_prop)
	{
		nbody_kernel::calc_gravity_accel_naive<<<grid, block>>>(n_obj, r, p, a);
	}
	else
	{
		nbody_kernel::calc_gravity_accel_naive_sym<<<grid, block>>>(n_obj, r, p, a);
	}
	CUDA_CHECK_ERROR();
	CUDA_SAFE_CALL(cudaEventRecord(t1));
	CUDA_SAFE_CALL(cudaEventSynchronize(t1));

	float dt = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&dt, t0, t1));

	return dt;
}

float gpu_calc_grav_accel_tile(uint32_t n_var, uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	set_kernel_launch_param(n_var, n_tpb, grid, block);
		
	printf(" grid: (%4u, %4u, %4u)\n", grid.x, grid.y, grid.z);
	printf("block: (%4u, %4u, %4u)\n", block.x, block.y, block.z);

	uint2_t sink   = {0, n_obj};
	uint2_t source = {0, n_obj};
	interaction_bound int_bound(sink, source);

	var3_t* r = (var3_t*)y_temp;
	var3_t* a = (var3_t*)(dy + 3*n_obj);
	nbp_t::param_t* p = (nbp_t::param_t*)d_p;

	cudaEvent_t t0, t1;
	CUDA_SAFE_CALL(cudaEventCreate(&t0));
	CUDA_SAFE_CALL(cudaEventCreate(&t1));

	CUDA_SAFE_CALL(cudaEventRecord(t0));
	// Clear the acceleration array: the += op can be used
	CUDA_SAFE_CALL(cudaMemset(a, 0, n_obj*sizeof(var3_t)));

	// Copy the velocities into dy
	// TODO: implement the asynchronous version of cudaMemcpy: Performace ??
	CUDA_SAFE_CALL(cudaMemcpy(dy, y_temp + 3*n_obj, 3*n_obj*sizeof(var_t), cudaMemcpyDeviceToDevice));

	nbody_kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var3_t)>>>(int_bound, n_tpb, r, p, a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(t1, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(t1));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, t0, t1));

	return elapsed_time;
}

void cpu_calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy, bool use_symm_prop)
{
	// Copy the velocities into dy
	memcpy(dy, y_temp + 3*n_obj, 3*n_obj*sizeof(var_t));

	var3_t* r = (var3_t*)y_temp;
	var3_t* a = (var3_t*)(dy + 3*n_obj);
	// Clear the acceleration array: the += op can be used
	memset(a, 0, 3*n_obj*sizeof(var_t));

	nbp_t::param_t* p = (nbp_t::param_t*)h_p;

	if (use_symm_prop)
	{
		for (uint32_t i = 0; i < n_obj; i++)
		{
			var3_t r_ij = {0, 0, 0};
			for (uint32_t j = i+1; j < n_obj; j++)
			{
				r_ij.x = r[j].x - r[i].x;
				r_ij.y = r[j].y - r[i].y;
				r_ij.z = r[j].z - r[i].z;

				var_t d2 = SQR(r_ij.x) + SQR(r_ij.y) + SQR(r_ij.z);
				var_t d = sqrt(d2);
				var_t d_3 = 1.0 / (d*d2);

				var_t s = p[j].mass * d_3;
				a[i].x += s * r_ij.x;
				a[i].y += s * r_ij.y;
				a[i].z += s * r_ij.z;

				s = p[i].mass * d_3;
				a[j].x -= s * r_ij.x;
				a[j].y -= s * r_ij.y;
				a[j].z -= s * r_ij.z;
			}
			a[i].x *= K2;
			a[i].y *= K2;
			a[i].z *= K2;
		}
	}
	else
	{
		for (uint32_t i = 0; i < n_obj; i++)
		{
			var3_t r_ij = {0, 0, 0};
			for (uint32_t j = 0; j < n_obj; j++)
			{
				if (i == j)
				{
					continue;
				}
				r_ij.x = r[j].x - r[i].x;
				r_ij.y = r[j].y - r[i].y;
				r_ij.z = r[j].z - r[i].z;

				var_t d2 = SQR(r_ij.x) + SQR(r_ij.y) + SQR(r_ij.z);
				var_t d = sqrt(d2);
				var_t d_3 = 1.0 / (d*d2);

				var_t s = p[j].mass * d_3;
				a[i].x += s * r_ij.x;
				a[i].y += s * r_ij.y;
				a[i].z += s * r_ij.z;
			}
			a[i].x *= K2;
			a[i].y *= K2;
			a[i].z *= K2;
		}
	}
}

void parse(int argc, const char** argv, uint32_t* n_obj)
{
	int i = 1;

	if (1 >= argc)
	{
		throw std::string("Missing command line arguments. For help use -h.");
	}

	while (i < argc)
	{
		std::string p = argv[i];
		if (     p == "-n")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw std::string("Invalid number at: " + p);
			}
			*n_obj = atoi(argv[i]);
		}
		else
		{
			throw std::string("Invalid switch on command line: " + p + ".");
		}
		i++;
	}
}

int main(int argc, const char *argv[])
{
	var_t* h_y = NULL;
	var_t* h_dy = NULL;
	var_t* h_dy0 = NULL;

	var_t* d_y = NULL;
	var_t* d_dy = NULL;

	uint32_t n_var = 0;
	uint32_t n_par = 0;

	try
	{
		// n_obj is a global variable
		parse(argc, argv, &n_obj);
		n_var = 6 * n_obj;
		n_par = 1 * n_obj;

		// Allocate HOST memory
		ALLOCATE_HOST_VECTOR((void**)&h_y,   n_var * sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_dy,  n_var * sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_dy0, n_var * sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&h_p,   n_par * sizeof(var_t));

		// Allocate DEVICE memory
		ALLOCATE_DEVICE_VECTOR((void**)&d_y,  n_var * sizeof(var_t));
		ALLOCATE_DEVICE_VECTOR((void**)&d_dy, n_var * sizeof(var_t));
		ALLOCATE_DEVICE_VECTOR((void**)&d_p,  n_par * sizeof(var_t));

		// Populate data
		srand(time(NULL));
		for (uint32_t i = 0; i < n_var; i++)
		{
			var_t r = (var_t)rand()/RAND_MAX;
			*(h_y + i) = r;
		}
		for (uint32_t i = 0; i < n_par; i++)
		{
			var_t r = (var_t)rand()/RAND_MAX;
			*(h_p + i) = 1;
		}

		CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, n_var * sizeof(var_t), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_p, h_p, n_par * sizeof(var_t), cudaMemcpyHostToDevice));

		var_t t0 = 0.0;
		uint16_t stage = 0;

		uint64_t T0 = GetTimeMs64();
		cpu_calc_dy(stage, t0, h_y, h_dy, false);
		uint64_t T1 = GetTimeMs64();
		var_t DT_CPU = ((var_t)(T1 - T0))/1000.0f;
		printf("CPU execution time: %16.4e [ms]\n", DT_CPU);

		T0 = GetTimeMs64();
		cpu_calc_dy(stage, t0, h_y, h_dy0, true);
		T1 = GetTimeMs64();
		DT_CPU = ((var_t)(T1 - T0))/1000.0f;
		printf("CPU execution time: %16.4e [ms]\n", DT_CPU);

		for (uint32_t j = 0; j < n_var; j++)
		{
			if (1.0e-15 < fabs(h_dy[j] - h_dy0[j]))
			{
				printf("Difference: j = %6u : %24.16e\n", j, h_dy[j] - h_dy0[j]);
			}
		}

		T0 = GetTimeMs64();
		float _DT_GPU = gpu_calc_dy(n_var, stage, t0, d_y, d_dy, false);
		T1 = GetTimeMs64();
		var_t DT_GPU = ((var_t)(T1 - T0))/1000.0f;
		printf("GPU execution time: %16.4e [ms]\n", DT_GPU);
		printf("GPU execution time: %16.4e [ms]\n", _DT_GPU);
		printf("%10u %16.4e %16.4e %16.4e %16.4e\n", n_obj, DT_CPU, DT_GPU, _DT_GPU, DT_CPU/_DT_GPU);

		// Copy down the data from the DEVICE
		CUDA_SAFE_CALL(cudaMemcpy(h_dy0, d_dy, n_var * sizeof(var_t), cudaMemcpyDeviceToHost));

		for (uint32_t j = 0; j < n_var; j++)
		{
			if (1.0e-15 < fabs(h_dy[j] - h_dy0[j]))
			{
				printf("Difference: j = %6u : %24.16e\n", j, h_dy[j] - h_dy0[j]);
			}
		}

		T0 = GetTimeMs64();
		_DT_GPU = gpu_calc_dy(n_var, stage, t0, d_y, d_dy, true);
		T1 = GetTimeMs64();
		DT_GPU = ((var_t)(T1 - T0))/1000.0f;
		printf("GPU execution time: %16.4e [ms]\n", DT_GPU);
		printf("GPU execution time: %16.4e [ms]\n", _DT_GPU);
		printf("%10u %16.4e %16.4e %16.4e %16.4e\n", n_obj, DT_CPU, DT_GPU, _DT_GPU, DT_CPU/_DT_GPU);

		// Copy down the data from the DEVICE
		CUDA_SAFE_CALL(cudaMemcpy(h_dy0, d_dy, n_var * sizeof(var_t), cudaMemcpyDeviceToHost));

		for (uint32_t j = 0; j < n_var; j++)
		{
			if (1.0e-15 < fabs(h_dy[j] - h_dy0[j]))
			{
				printf("Difference: j = %6u : %24.16e\n", j, h_dy[j] - h_dy0[j]);
			}
		}

		T0 = GetTimeMs64();
		_DT_GPU = gpu_calc_grav_accel_tile(n_var, stage, t0, d_y, d_dy);
		T1 = GetTimeMs64();
		DT_GPU = ((var_t)(T1 - T0))/1000.0f;
		printf("GPU execution time: %16.4e [ms]\n", DT_GPU);
		printf("GPU execution time: %16.4e [ms]\n", _DT_GPU);
		printf("%10u %16.4e %16.4e %16.4e %16.4e\n", n_obj, DT_CPU, DT_GPU, _DT_GPU, DT_CPU/_DT_GPU);

		// Copy down the data from the DEVICE
		CUDA_SAFE_CALL(cudaMemcpy(h_dy0, d_dy, n_var * sizeof(var_t), cudaMemcpyDeviceToHost));

		for (uint32_t j = 0; j < n_var; j++)
		{
			if (1.0e-15 < fabs(h_dy[j] - h_dy0[j]))
			{
				printf("Difference: j = %6u : %24.16e\n", j, h_dy[j] - h_dy0[j]);
			}
		}

		FREE_HOST_VECTOR((void**)&h_y  );
		FREE_HOST_VECTOR((void**)&h_dy );
		FREE_HOST_VECTOR((void**)&h_dy0);
		FREE_HOST_VECTOR((void**)&h_p  );

		FREE_DEVICE_VECTOR((void**)&d_y );
		FREE_DEVICE_VECTOR((void**)&d_dy);
		FREE_DEVICE_VECTOR((void**)&d_p );
	}
	catch (const std::string& msg)
	{
		std::cerr << "Error: " << msg << std::endl;
	}
	std::cout << "Gravitational interaction computations done.\n";

	return 0;
}

#endif

#if 0
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand, malloc       */
#include <time.h>       /* time                      */

#include <iostream>
#include <string>

#include "constants.h"
#include "type.h"
#include "redutil2.h"

using namespace std;
using namespace redutil2;

int comp_value(var_t v1, var_t v2, var_t tol, char* lpad, char* text)
{
	int result = 0;

	var_t d = fabs(v1 - v2);
	if (tol < d)
	{
		printf("%s%s = %25.15lg\n", lpad, text, d);
		result = 1;
	}

	return result;
}

int comp_oe(orbelem_t &oe1, orbelem_t& oe2, var_t tol, char* lpad)
{
	int result = comp_value(oe1.sma, oe2.sma, tol, lpad, "Abs(Delta(sma ))");
	result += comp_value(oe1.ecc, oe2.ecc, tol, lpad, "Abs(Delta(ecc ))");
	result += comp_value(oe1.inc, oe2.inc, tol, lpad, "Abs(Delta(inc ))");
	result += comp_value(oe1.peri, oe2.peri, tol, lpad, "Abs(Delta(peri))");
	result += comp_value(oe1.node, oe2.node, tol, lpad, "Abs(Delta(node))");
	result += comp_value(oe1.mean, oe2.mean, tol, lpad, "Abs(Delta(mean))");
	return result;
}

int comp_2D_vectors(var2_t &v1, var2_t &v2, var_t tol, char* lpad)
{
	int result = comp_value(v1.x, v2.x, tol, lpad, "Abs(Delta(v1.x - v2.x))");
	result += comp_value(v1.y, v2.y, tol, lpad, "Abs(Delta(v1.y - v2.y))");
	return result;
}

var_t random(var_t x0, var_t x1)
{
	return (x0 + ((var_t)rand() / RAND_MAX) * (x1 - x0));
}

void test_calc_ephemeris()
{
	// Test calculate phase from orbital elements and vice versa
	{
		const char func_name[] = "calc_phase";
		char lpad[] = "        ";
		/*
		 * The units are:
		 *     Unit name         | Unit symbol | Quantity name
		 *     -----------------------------------------------
		 *     Astronomical unit |          AU | length
		 *     Solar mass        |           S | mass
		 *     Mean solar day    |           D | time
		 */

		srand((unsigned int)time(NULL));
		// parameter of the problem
		tbp_t::param_t p;
            
		// Set the parameter of the problem
		p.mu = constants::Gauss2 * (1.0 + 1.0);
		// Set the initial orbital elements
		orbelem_t oe1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		orbelem_t oe2 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};

		var_t tol = 1.0e-14;
		for (int i = 0; i < 100; i++)
		{
			oe1.sma = random(0.1, 10.0);
			oe1.ecc = random(0.0, 0.8);
			oe1.inc = random(0.0, PI);
			oe1.peri =random(0.0, TWOPI);
			oe1.node =random(0.0, TWOPI);
			oe1.mean =random(0.0, TWOPI);
			// Calculate the position and velocity vectors from orbital elements
			tools::calc_phase(p.mu, &oe1, &r0, &v0);
			// Calculate the orbital elements from position and velocity vectors
			tools::calc_oe(p.mu, &r0, &v0, &oe2);
	
			int ret_val = comp_oe(oe1, oe2, tol, lpad);
			if (0 < ret_val)
			{
				printf("    TEST '%s' failed with tolerance level: %g\n", func_name, tol);
			}
			else
			{
				printf("    TEST '%s' passed with tolerance level: %g\n", func_name, tol);
			}
		}
	} /* Test calc_phase() and calc_oe() functions */
}

void test_rtbp2d_calc_energy()
{
	// Test tools::tbp::calc_integral() and tools::rtbp2D::calc_integral() functions
	{
		const char func_name[] = "tools::tbp::calc_integral";
		char lpad[] = "        ";

	    /*
	     * The units are:
	     *     Unit name         | Unit symbol | Quantity name
	     *     -----------------------------------------------
	     *     Astronomical unit |          AU | length
	     *     Solar mass        |           S | mass
	     *     Mean solar day    |           D | time
	     */

		srand(0);

		orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};

		var_t mu = constants::Gauss2*(1.0 + 1.0);
		var_t tol = 1.0e-15;
		for (int i = 0; i < 10; i++)
		{
			// Set the initial orbital elements
			oe.sma  = random(0.1, 10.0);
			oe.ecc  = random(0.0, 0.8);
			oe.inc  = 0.0;
			oe.peri = random(0.0, TWOPI);
			oe.node = 0.0;
			oe.mean = random(0.0, TWOPI);
			// Calculate the position and velocity vectors from orbital elements
			tools::calc_phase(mu, &oe, &r0, &v0);

			// Set the starting coordinate and velocity vectors
			var2_t r  = {r0.x, r0.y};
			var2_t v  = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};
			tools::rtbp2D::transform_x2u(r, u);
			tools::rtbp2D::transform_xd2up(u, v, up);

			var_t hs = tools::tbp::calc_integral(mu, r, v);
			var_t hr = tools::rtbp2D::calc_integral(mu, u, up);

			printf("    hs = %25.15le\n", hs);
			printf("    hr = %25.15le\n", hr);
		}

		// Calculate the energy along a Kepler-orbit
		oe.sma  = 1.5;
		oe.ecc  = 0.8;
		oe.inc  = 0.0;
		oe.peri = 0.0;
		oe.node = 0.0;
		oe.mean = 0.0;
		do
		{
			tools::calc_phase(mu, &oe, &r0, &v0);
			var2_t r  = {r0.x, r0.y};
			var2_t v  = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};
			tools::rtbp2D::transform_x2u(r, u);
			tools::rtbp2D::transform_xd2up(u, v, up);

			var_t hs = tools::tbp::calc_integral(mu, r, v);
			var_t hr = tools::rtbp2D::calc_integral(mu, u, up);
			printf("%25.15le %25.15le %25.15le\n", oe.mean, hs, hr);

			oe.mean += 1.0 * constants::DegreeToRadian;
		} while (oe.mean <= TWOPI);
	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}

void test_rtbp2d_transform()
{
	// Test square (section lines)
	{
		var_t d = 0.01;
		// Q4 -> Q1
		var2_t x = {0.5, -0.5};
		var2_t u  = {0, 0};
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.y += d;
		} while (0.5 >= x.y);
		// Q1 -> Q2
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.x -= d;
		} while (-0.5 <= x.x);
		// Q2 -> Q3
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.y -= d;
		} while (-0.5 <= x.y);
		// Q3 -> Q4
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.x += d;
		} while (0.5 >= x.x);
	}

	return;

	// Test ellipse
	{
		const char func_name[] = "tools::rtbp2D::transform___";
		char lpad[] = "        ";

	    /*
	     * The units are:
	     *     Unit name         | Unit symbol | Quantity name
	     *     -----------------------------------------------
	     *     Astronomical unit |          AU | length
	     *     Solar mass        |           S | mass
	     *     Mean solar day    |           D | time
	     */

		srand(0);

		const var_t mu = constants::Gauss2*(1.0 + 1.0);
		orbelem_t oe = {0.5, 0.8, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};
		int i = 0;
		do
		{
			oe.mean = i * constants::DegreeToRadian;
			tools::calc_phase(mu, &oe, &r0, &v0);
			var2_t x  = {r0.x, r0.y};
			var2_t xd = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};

			tools::rtbp2D::transform_x2u(x, u);
			tools::rtbp2D::transform_xd2up(u, xd, up);
			x.x  = x.y  = 0.0;
			xd.x = xd.y = 0.0;

			tools::rtbp2D::transform_u2x(u, x);
			tools::rtbp2D::transform_up2xd(u, up, xd);
			// Compare the original position and velocitiy vectors with the calculated ones
			{
				var_t tol = 1.0e-15;
				var2_t x0  = {r0.x, r0.y};
				var2_t x0d = {v0.x, v0.y};
				comp_2D_vectors(x0, x, tol, lpad);
				comp_2D_vectors(x0d, xd, tol, lpad);
			}

			printf("%23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le\n", oe.mean, x.x, x.y, u.x, u.y, xd.x, xd.y, up.x, up.y);
			if (0 < i && 0 == (i+1) % 90)
			{
				printf("\n");
			}
			i++;
		} while (360 > i);
	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}

void test_calc_lin_comb()
{
	// Test calculate linear combination of vectors
	{
		const char func_name[] = "calc_lin_comb";
		char lpad[] = "        ";

		uint32_t n_var = 2;
		uint16_t n_vct = 3;

		var_t* a = NULL;
		var_t* b = NULL;
		var_t** c = NULL;
		var_t* coeff = NULL;

		ALLOCATE_HOST_VECTOR((void**)&(a), n_var*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&(b), n_var*sizeof(var_t));
		
		ALLOCATE_HOST_VECTOR((void**)&c, n_vct*sizeof(var_t*));
		for (uint16_t i = 0; i < n_vct; i++)
		{
			ALLOCATE_HOST_VECTOR((void**)&(c[i]), n_var*sizeof(var_t));
		}
		ALLOCATE_HOST_VECTOR((void**)&coeff, n_vct*sizeof(var_t));
	
		// Populate vectors
		memset(a, 0, n_var*sizeof(var_t));
		for (int i = 0; i < n_var; i++)
		{
			b[i] = -(i+1);
		}
		for (uint32_t i = 0; i < n_vct; i++)
		{
			for (uint32_t j = 0; j < n_var; j++)
			{
				c[i][j] = i+j+1;
			}
		}
		for (int i = 0; i < n_vct; i++)
		{
			coeff[i] = 10*i;
		}

		printf("The data in the vectors:\n");
		printf("a:\n");
		print_array("", n_var, a, PROC_UNIT_CPU);
		printf("b:\n");
		print_array("", n_var, b, PROC_UNIT_CPU);
		for (uint32_t i = 0; i < n_vct; i++)
		{
			printf("c[%d]:\n", i);
			print_array("", n_var, c[i], PROC_UNIT_CPU);
		}
		printf("The coefficients:\n");
		print_array("", n_vct, coeff, PROC_UNIT_CPU);

		// Calculate the linear combination of the vectors
		tools::calc_lin_comb(a, c, coeff, n_vct, n_var);
		printf("The linear combination of the vectors:\n");
		printf("a:\n");
		print_array("", n_var, a, PROC_UNIT_CPU);

		// Calculate the special case of linear combination of the vectors
		tools::calc_lin_comb_s(a, b, c, coeff, n_vct, n_var);
		printf("The special linear combination of the vectors:\n");
		printf("a:\n");
		print_array("", n_var, a, PROC_UNIT_CPU);

		FREE_HOST_VECTOR((void **)&(coeff));
		for (uint16_t i = 0; i < n_vct; i++)
		{
			FREE_HOST_VECTOR((void **)&(c[i]));
		}
		FREE_HOST_VECTOR((void **)&(c));
		FREE_HOST_VECTOR((void **)&(b));
		FREE_HOST_VECTOR((void **)&(a));
	}	

	// Test calculate linear combination of two vectors
	{
		const char func_name[] = "calc_lin_comb_s";
		char lpad[] = "        ";

		uint32_t n_var = 2;

		var_t* a = NULL;
		var_t* b = NULL;
		var_t* c = NULL;
		var_t f = 3;

		ALLOCATE_HOST_VECTOR((void**)&(a), n_var*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&(b), n_var*sizeof(var_t));
		ALLOCATE_HOST_VECTOR((void**)&(c), n_var*sizeof(var_t));	

		// Populate vectors
		memset(a, 0, n_var*sizeof(var_t));
		for (int i = 0; i < n_var; i++)
		{
			b[i] = -(i+1);
			c[i] = i+1;
		}

		printf("The data in the vectors:\n");
		printf("a:\n");
		print_array("", n_var, a, PROC_UNIT_CPU);
		printf("b:\n");
		print_array("", n_var, b, PROC_UNIT_CPU);
		printf("c:\n");
		print_array("", n_var, c, PROC_UNIT_CPU);
		printf("The coefficient:\n");
		printf("%5e\n", f);

		// Calculate the special case of linear combination of the vectors
		tools::calc_lin_comb_s(a, b, c, f, n_var);
		printf("The special linear combination of two vectors:\n");
		printf("a:\n");
		print_array("", n_var, a, PROC_UNIT_CPU);

		FREE_HOST_VECTOR((void **)&(c));
		FREE_HOST_VECTOR((void **)&(b));
		FREE_HOST_VECTOR((void **)&(a));
	}	
}

/*

cd 'C:\Work\red.cuda.Results\v2.0\Test_Copy\rtbp2D\Test_transform
a=1.0
p [-1:1][-1:1]'e_0.0_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.0_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q4.txt' u 6:7 w l, '' u 8:9 w l

a=1.0
p [-1:1][-1:1]'e_0.2_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.2_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q4.txt' u 6:7 w l, '' u 8:9 w l

a=1.0
p [-1:1][-1:1]'e_0.8_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.8_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q4.txt' u 6:7 w l, '' u 8:9 w l
*/
int main()
{
	try
	{
		//test_calc_ephemeris();
		//test_rtbp2d_trans();
		//test_rtbp2d_transform();
		//test_rtbp2d_calc_energy();
		test_calc_lin_comb();
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

    return 0;
}

#endif
