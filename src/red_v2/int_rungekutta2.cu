#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_rungekutta2.h"

#include "redutil2.h"
#include "macro.h"

using namespace redutil2;

// The Runge-Kutta matrix
var_t int_rungekutta2::a[] = 
{ 
	0.0,     0.0, 
	1.0/2.0, 0.0
};
// weights
var_t int_rungekutta2::b[] = { 0.0, 1.0     };
// nodes
var_t int_rungekutta2::c[] = { 0.0, 1.0/2.0 };
// These arrays will contain the stepsize multiplied by the constants
var_t int_rungekutta2::aa[ sizeof(int_rungekutta2::a ) / sizeof(var_t)];
var_t int_rungekutta2::bb[ sizeof(int_rungekutta2::b ) / sizeof(var_t)];
var_t int_rungekutta2::cc[ sizeof(int_rungekutta2::c ) / sizeof(var_t)];

__constant__ var_t dc_a[ sizeof(int_rungekutta2::a ) / sizeof(var_t)];
__constant__ var_t dc_b[ sizeof(int_rungekutta2::b ) / sizeof(var_t)];
__constant__ var_t dc_c[ sizeof(int_rungekutta2::c ) / sizeof(var_t)];


namespace rk2_kernel
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

} /* namespace rk2_kernel */

int_rungekutta2::int_rungekutta2(ode& f, var_t dt, comp_dev_t comp_dev) :
	integrator(f, dt, false, 0.0, 2, comp_dev)
{
	name    = "Runge-Kutta2";
	n_order = 2;

	if (COMP_DEV_GPU == comp_dev)
	{
		redutil2::copy_constant_to_device(dc_a, a, sizeof(a));
		redutil2::copy_constant_to_device(dc_b, b, sizeof(b));
		redutil2::copy_constant_to_device(dc_c, c, sizeof(c));
	}
}

int_rungekutta2::~int_rungekutta2()
{}

void int_rungekutta2::calc_y_np1()
{
	if (COMP_DEV_GPU == comp_dev)
	{
		rk2_kernel::calc_lin_comb<<<grid, block>>>(f.yout, f.y, dt_try, k[1], f.n_var);
		CUDA_CHECK_ERROR();
	}
	else
	{
		tools::calc_lin_comb_s(f.yout, f.y, k[1], dt_try, f.n_var);
	}
}

void int_rungekutta2::calc_ytemp(uint16_t stage)
{
	if (COMP_DEV_GPU == comp_dev)
	{
	}
	else
	{
		var_t* coeff = aa + stage * n_stage;
		tools::calc_lin_comb_s(ytemp, f.y, k, coeff, stage, f.n_var);
	}
}

var_t int_rungekutta2::step()
{
	static const uint16_t n_aa = sizeof(int_rungekutta2::aa) / sizeof(var_t);

	if (COMP_DEV_GPU == comp_dev)
	{
		redutil2::set_kernel_launch_param(f.n_var, THREADS_PER_BLOCK, grid, block);
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.y, k[stage]);

	// TODO: check if this speeds up the app or not!
	// Compute in advance the dt_try * coefficients to save n_var multiplication per stage
	for (uint16_t i = 0; i < n_aa; i++)
	{
		aa[i] = dt_try * a[i];
	}

	stage = 1;
	t = f.t + c[stage] * dt_try;
	calc_ytemp(stage);
	f.calc_dy(stage, t, ytemp, k[stage]);

	calc_y_np1();

	dt_did = dt_try;

	update_counters(1);

	f.tout = t = f.t + dt_did;
	f.swap();

	return dt_did;
}
