#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class int_rungekutta7 : public integrator
{
public:
	static var_t a[], h_a[];
	static var_t b[];
	static var_t bh[], h_bh;
	static var_t c[];
	static uint16_t a_idx[];

	int_rungekutta7(ode& f, bool adaptive, var_t tolerance, comp_dev_t comp_dev);
	~int_rungekutta7();

	void allocate_Butcher_tableau();
	void deallocate_Butcher_tableau();
	var_t step();

private:
	var_t *d_a;
	var_t *d_bh;

	void calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	void calc_error(uint32_t n);

	void cpu_calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	void cpu_calc_error(uint32_t n);
};
