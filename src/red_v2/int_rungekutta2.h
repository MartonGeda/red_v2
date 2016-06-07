#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class int_rungekutta2 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t c[];

	int_rungekutta2(ode& f, var_t dt, comp_dev_t comp_dev);
	~int_rungekutta2();

	var_t step();

private:
	void cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n);
	void calc_ytemp(uint16_t stage);
	void calc_y_np1();
};
