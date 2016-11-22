#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class int_rungekutta5 : public integrator
{
public:
	static var_t a[], _a[];
	static var_t bh[], _bh[];
	static var_t c[];

	int_rungekutta5(ode& f, bool adaptive, var_t tolerance, comp_dev_t comp_dev);
	~int_rungekutta5();

	var_t step();

private:
	void calc_ytemp(uint16_t stage);
	void calc_y_np1();

	void calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	void calc_error(uint32_t n);

	void cpu_calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	void cpu_calc_error(uint32_t n);
};
