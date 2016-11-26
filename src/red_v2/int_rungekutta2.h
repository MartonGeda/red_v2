#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class int_rungekutta2 : public integrator
{
public:
	static var_t a[], _a[];
	static var_t b[], _b[];
	static var_t c[], _c[];

	int_rungekutta2(ode& f, comp_dev_t comp_dev);
	~int_rungekutta2();

	var_t step();

private:
	void calc_ytemp(uint16_t stage);
	void calc_y_np1();
};
