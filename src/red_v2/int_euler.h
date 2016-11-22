#pragma once

#include "integrator.h"

#include "type.h"

class ode;

class euler : public integrator
{
public:
	euler(ode& f, comp_dev_t comp_dev);
	~euler();

	var_t step();

private:
	void calc_y_np1();
};
