#include <string>

#include "ode.h"
#include "redutil2.h"

using namespace redutil2;

ode::ode(uint16_t n_dim, uint32_t n_obj, uint16_t n_vpo, uint16_t n_ppo, comp_dev_t comp_dev) :
	n_dim(n_dim),
	n_obj(n_obj),
	n_vpo(n_vpo),
	n_ppo(n_ppo),
	comp_dev(comp_dev)
{
	initialize();

	n_var  = n_obj * n_vpo;
	n_par  = n_obj * n_ppo;
	allocate_storage(n_var, n_par);
	create_aliases();
}

ode::ode(uint16_t n_dim, uint32_t n_obj, uint16_t n_vpo, uint16_t n_ppo, uint32_t n_var, uint32_t n_par, comp_dev_t comp_dev) :
	n_dim(n_dim),
	n_obj(n_obj),
	n_vpo(n_vpo),
	n_ppo(n_ppo),
	n_var(n_var),
	n_par(n_par),
	comp_dev(comp_dev)
{
	initialize();

	allocate_storage(n_var, n_par);
	create_aliases();
}

ode::~ode()
{
	deallocate_storage();
}

void ode::initialize()
{
	t      = 0.0;
	tout   = 0.0;
	dt     = 0.0;

	h_y    = 0x0;
	h_yout = 0x0;

	d_y	   = 0x0;
	d_yout = 0x0;

	y	   = 0x0;
	yout   = 0x0;
	
	h_p    = 0x0;
	d_p	   = 0x0;
	p	   = 0x0;

	n_tpb  = 1;

	var3_t zero = {0, 0, 0};
	integral.h0 = integral.h = 0;
	integral.c0 = integral.c = zero;
	integral.R0 = integral.R = zero;
	integral.V0 = integral.V = zero;
}

void ode::allocate_storage(uint32_t n_var, uint32_t n_par)
{
	allocate_host_storage(n_var, n_par);
	if (COMP_DEV_GPU == comp_dev)
	{
		allocate_device_storage(n_var, n_par);
	}
}

void ode::allocate_host_storage(uint32_t n_var, uint32_t n_par)
{
	ALLOCATE_HOST_VECTOR((void**)&(h_y),    n_var * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_yout), n_var * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_p),    n_par * sizeof(var_t));
}

void ode::allocate_device_storage(uint32_t n_var, uint32_t n_par)
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_y),    n_var * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_yout), n_var * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_p),    n_par * sizeof(var_t));
}

void ode::deallocate_storage()
{
	deallocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void ode::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_y));
	FREE_HOST_VECTOR((void **)&(h_yout));
	FREE_HOST_VECTOR((void **)&(h_p));
}

void ode::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(d_y));
	FREE_DEVICE_VECTOR((void **)&(d_yout));
	FREE_DEVICE_VECTOR((void **)&(d_p));
}

// Date of creation: 2016.08.03.
// Last edited: 
// Status: Not tested
void ode::create_aliases()
{
	switch (comp_dev)
	{
	case COMP_DEV_CPU:
		y    = h_y;
		yout = h_yout;
		p    = h_p;
		break;
	case COMP_DEV_GPU:
		y    = d_y;
		yout = d_yout;
		p    = d_p;
		break;
	default:
		throw std::string("Parameter 'comp_dev' is out of range.");
	}
}

void ode::copy_vars(copy_direction_t dir)
{
	switch (dir)
	{
	case COPY_DIRECTION_TO_DEVICE:
		copy_vector_to_device(d_y, h_y, n_var*sizeof(var_t));
		break;
	case COPY_DIRECTION_TO_HOST:
		copy_vector_to_device(h_y, d_y, n_var*sizeof(var_t));
	default:
		throw std::string("Parameter 'dir' is out of range.");
		break;
	}
}

void ode::copy_params(copy_direction_t dir)
{
	switch (dir)
	{
	case COPY_DIRECTION_TO_DEVICE:
		copy_vector_to_device(d_p, h_p, n_par*sizeof(var_t));
		break;
	case COPY_DIRECTION_TO_HOST:
		copy_vector_to_device(h_p, d_p, n_par*sizeof(var_t));
	default:
		throw std::string("Parameter 'dir' is out of range.");
		break;
	}
}

void ode::swap()
{
	std::swap(t, tout);

	std::swap(y, yout);
	std::swap(h_y, h_yout);
	std::swap(d_y, d_yout);
}
