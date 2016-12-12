#pragma once

#include "integrator.h"
#include "type.h"

class ode;

class int_hermite4 : public integrator
{
public:
    int_hermite4(ode& f, bool adaptive, var_t tolerance, comp_dev_t comp_dev);
    ~int_hermite4();

	void allocate_Butcher_tableau();
	void deallocate_Butcher_tableau();
	void check_Butcher_tableau();
	var_t step();

private:

    void predict();
    void correct();
};