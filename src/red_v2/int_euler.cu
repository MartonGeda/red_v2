#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_euler.h"

#include "redutil2.h"
#include "macro.h"

using namespace redutil2;

namespace euler_kernel
{
// a_i = b_i + F * c_i
static __global__
	void calc_lin_comb(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		a[tid] = b[tid] + F * c[tid];
		tid += stride;
	}
}
} /* namespace euler_kernel */

euler::euler(ode& f, var_t dt, comp_dev_t comp_dev) :
	integrator(f, dt, false, 0.0, 1, comp_dev)
{
	name    = "Euler";
	n_order = 1;
}

euler::~euler()
{}

void euler::calc_lin_comb(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
{
	for (uint32_t tid = 0; tid < n; tid++)
	{
		a[tid] = b[tid] + F * c[tid];
	}
}

void euler::calc_y_np1()
{
	if (COMP_DEV_GPU == comp_dev)
	{
		euler_kernel::calc_lin_comb<<<grid, block>>>(f.yout, f.y, dt_try, k[0], f.n_var);
		CUDA_CHECK_ERROR();
	}
	else
	{
		tools::calc_lin_comb_s(f.yout, f.y, k[0], dt_try, f.n_var);
		//calc_lin_comb(f.yout, f.y, dt_try, k[0], f.n_var);
	}
}

var_t euler::step()
{
	if (COMP_DEV_GPU == comp_dev)
	{
		redutil2::set_kernel_launch_param(f.n_var, THREADS_PER_BLOCK, grid, block);
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.y, k[stage]);

	calc_y_np1();

	dt_did = dt_try;

	update_counters(1);

	f.tout = t = f.t + dt_did;
	f.swap();

	return dt_did;
}
