#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_euler.h"

#include "redutil2.h"
#include "macro.h"

using namespace redutil2;

euler::euler(ode& f, comp_dev_t comp_dev) :
	integrator(f, false, 0.0, 1, comp_dev)
{
	name    = "Euler";
	n_order = 1;
}

euler::~euler()
{}

void euler::calc_y_np1()
{
	static uint32_t n_var = 0;
	bool benchmark = true;

	if (n_var != f.n_var)
	{
		benchmark = true;
		n_var = f.n_var;
	}
	else
	{
		benchmark = false;
	}

	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		gpu_calc_lin_comb_s(f.yout, f.y, k[0], dt_try, f.n_var, comp_dev.id_dev, benchmark);
		CUDA_CHECK_ERROR();
	}
	else
	{
		tools::calc_lin_comb_s(f.yout, f.y, k[0], dt_try, f.n_var);
	}
}

var_t euler::step()
{
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
