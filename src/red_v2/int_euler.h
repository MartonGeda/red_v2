#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class euler : public integrator
{
public:
	euler(ode& f, var_t dt, comp_dev_t comp_dev);
	~euler();

	var_t step();

private:
	void calc_lin_comb(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n);
	void calc_y_np1();
};
