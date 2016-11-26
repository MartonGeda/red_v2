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
	0.0,     0.0,      // y = yn          -> k1
	1.0/2.0, 0.0       // y = yn + h*k1
};
// weights
var_t int_rungekutta2::b[] = { 0.0, 1.0     };
// nodes
var_t int_rungekutta2::c[] = { 0.0, 1.0/2.0 };
// These arrays will contain the stepsize multiplied by the constants
var_t int_rungekutta2::_a[ sizeof(int_rungekutta2::a ) / sizeof(var_t)];
var_t int_rungekutta2::_b[ sizeof(int_rungekutta2::b ) / sizeof(var_t)];
var_t int_rungekutta2::_c[ sizeof(int_rungekutta2::c ) / sizeof(var_t)];

__constant__ var_t dc_a[ sizeof(int_rungekutta2::a ) / sizeof(var_t)];


int_rungekutta2::int_rungekutta2(ode& f, comp_dev_t comp_dev) :
	integrator(f, false, 0.0, 2, comp_dev)
{
	name    = "Runge-Kutta2";
	n_order = 2;
}

int_rungekutta2::~int_rungekutta2()
{ }

void int_rungekutta2::calc_ytemp(uint16_t stage)
{
	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		var_t* coeff = dc_a + stage * n_stage;
		gpu_calc_lin_comb_s(ytemp, f.y, d_k, coeff, stage, f.n_var, comp_dev.id_dev, optimize);
	}
	else
	{
		var_t* coeff = _a + stage * n_stage;
		tools::calc_lin_comb_s(ytemp, f.y, h_k, coeff, stage, f.n_var);
	}
}

void int_rungekutta2::calc_y_np1()
{
	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		gpu_calc_lin_comb_s(f.yout, f.y, k[1], dt_try, f.n_var, comp_dev.id_dev, optimize);
	}
	else
	{
		tools::calc_lin_comb_s(f.yout, f.y, k[1], dt_try, f.n_var);
	}
}

var_t int_rungekutta2::step()
{
	static const uint16_t n_a = sizeof(int_rungekutta2::a) / sizeof(var_t);
	static uint32_t n_var = 0;

    if (n_var != f.n_var)
	{
		optimize = true;
		n_var = f.n_var;
	}
	else
	{
		optimize = false;
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.y, k[stage]);

	// TODO: check if this speeds up the app or not!
	// Compute in advance the dt_try * coefficients to save n_var multiplication per stage
	for (uint16_t i = 0; i < n_a; i++)
	{
		_a[i] = dt_try * a[i];
	}
	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		redutil2::copy_constant_to_device(dc_a, _a, sizeof(_a));
	}

	stage = 1;
	t = f.t + c[stage] * dt_try;
	calc_ytemp(stage);
	f.calc_dy(stage, t, ytemp, k[stage]);

	calc_y_np1();

	dt_did = dt_try;
	f.tout = t = f.t + dt_did;
	f.swap();

    update_counters(1);

	return dt_did;
}
