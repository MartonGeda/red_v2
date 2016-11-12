#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"

#include "integrator.h"
#include "ode.h"
#include "macro.h"

#include "redutil2.h"

using namespace std;
using namespace redutil2;

integrator::integrator(ode& f, var_t dt, bool adaptive, var_t tolerance, uint16_t n_stage, comp_dev_t comp_dev) : 
	f(f),
	dt_try(dt),
	adaptive(adaptive),
	tolerance(tolerance),
	n_stage(n_stage),
	comp_dev(comp_dev)
{
	initialize();

	allocate_storage(f.n_var);
	create_aliases();
}

integrator::~integrator()
{
	deallocate_storage();
}

void integrator::initialize()
{
	t             = f.t;
	dt_did        = 0.0;

	//d_ck          = NULL;
	h_k           = NULL;
	d_k           = NULL;
	k             = NULL;

	h_ytemp       = NULL;
	d_ytemp       = NULL;
	ytemp         = NULL;

	h_err         = NULL;
	d_err         = NULL;
	err           = NULL;

	max_iter      = 100;
	dt_min        = 1.0e10;

	n_tried_step  = 0;
	n_passed_step = 0;
	n_failed_step = 0;
}

void integrator::allocate_storage(uint32_t n_var)
{
	allocate_host_storage(n_var);
	if (COMP_DEV_GPU == comp_dev)
	{
		allocate_device_storage(n_var);
	}
}

void integrator::allocate_host_storage(uint32_t n_var)
{
	h_k = (var_t**)malloc(n_stage*sizeof(var_t*));
	if (NULL == h_k)
	{
		throw string("Host memory allocation failed.");
	}
	memset(h_k, 0, n_stage*sizeof(var_t*));

	k = (var_t**)malloc(n_stage*sizeof(var_t*));
	if (NULL == k)
	{
		throw string("Host memory allocation failed.");
	}
	memset(k, 0, n_stage*sizeof(var_t*));

	for (uint16_t i = 0; i < n_stage; i++)
	{
		ALLOCATE_HOST_VECTOR((void**)(h_k + i), n_var*sizeof(var_t));
	}
	ALLOCATE_HOST_VECTOR((void**)&(h_ytemp), n_var*sizeof(var_t));
	if (adaptive)
	{
		ALLOCATE_HOST_VECTOR((void**)&(h_err), n_var*sizeof(var_t));
	}
}

void integrator::allocate_device_storage(uint32_t n_var)
{
	CUDA_SAFE_CALL(cudaMalloc((void**)d_k, n_stage*sizeof(var_t*)));
	if (NULL == d_k)
	{
		throw string("Device memory allocation failed.");
	}
	// Clear memory 
	CUDA_SAFE_CALL(cudaMemset((void**)d_k, 0, n_stage*sizeof(var_t)));

	//ALLOCATE_DEVICE_VECTOR((void**)&d_k, n_stage*sizeof(var_t*));
	//ALLOCATE_DEVICE_VECTOR((void**)&d_ck, n_stage*sizeof(var_t*));
	for (uint16_t i = 0; i < n_stage; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_k[i]), n_var*sizeof(var_t));
		//copy_vector_to_device((void*)&d_ck[i], &d_k[i], sizeof(var_t*));
	}
	ALLOCATE_DEVICE_VECTOR((void**)&(d_ytemp), n_var*sizeof(var_t));
	if (adaptive)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_err), n_var*sizeof(var_t));
	}
}

void integrator::deallocate_storage()
{
	deallocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void integrator::deallocate_host_storage()
{
	for (uint16_t i = 0; i < n_stage; i++)
	{
		FREE_HOST_VECTOR((void **)(h_k + i));
	}
	free(h_k); h_k = NULL;
	free(k);   k = NULL;

	FREE_HOST_VECTOR((void **)&(h_ytemp));
	if (adaptive)
	{
		FREE_HOST_VECTOR((void **)&(h_err));
	}
}

void integrator::deallocate_device_storage()
{
	//FREE_DEVICE_VECTOR((void **)&(d_ck));
	for (uint16_t i = 0; i < n_stage; i++)
	{
		FREE_DEVICE_VECTOR((void **)&(d_k[i]));
	}
	FREE_DEVICE_VECTOR((void **)&(d_k));

	FREE_DEVICE_VECTOR((void **)&(d_ytemp));
	if (adaptive)
	{
		FREE_DEVICE_VECTOR((void **)&(d_err));
	}
}

// Date of creation: 2016.08.02.
// Last edited: 
// Status: Not tested
void integrator::create_aliases()
{
	switch (comp_dev)
	{
	case COMP_DEV_CPU:
		ytemp = h_ytemp;
		for (int r = 0; r < n_stage; r++) 
		{
			k[r] = h_k[r];
		}
		if (adaptive)
		{
			err = h_err;
		}
		break;
	case COMP_DEV_GPU:
		ytemp = d_ytemp;
		for (int r = 0; r < n_stage; r++) 
		{
			k[r] = d_k[r];
		}
		if (adaptive)
		{
			err = d_err;
		}
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}
}

void integrator::set_computing_device(comp_dev_t device)
{
	// If the execution is already on the requested device than nothing to do
	if (this->comp_dev == device)
	{
		return;
	}
	// TODO: implement

	//int n_body = ppd->n_bodies->get_n_total_playing();

	//switch (device)
	//{
	//case COMP_DEV_CPU:
	//	deallocate_device_storage();
	//	break;
	//case COMP_DEV_GPU:
	//	allocate_device_storage(n_body);
	//	break;
	//default:
	//	throw string("Parameter 'device' is out of range.");
	//}

	//this->comp_dev = device;
	//create_aliases();
	//f->set_computing_device(device);
}


var_t integrator::get_max_error(uint32_t n_var)
{
	var_t max_err = 0.0;

	if (COMP_DEV_GPU == comp_dev)
	{
		// Wrap raw pointer with a device_ptr
		thrust::device_ptr<var_t> d_ptr(d_err);
		// Use thrust to find the maximum element
		thrust::device_ptr<var_t> d_ptr_max = thrust::max_element(d_ptr, d_ptr + n_var);
		// Copy the max element from device memory to host memory
		cudaMemcpy((void*)&max_err, (void*)d_ptr_max.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	}
	else
	{
		for (uint32_t i = 0; i < n_var; i++)
		{
			if (max_err < fabs(h_err[i]))
			{
				max_err = fabs(h_err[i]);
			}
		}		
	}
	return (max_err);
}

void integrator::calc_dt_try(var_t max_err)
{
	if (1.0e-20 < max_err)
	{
		dt_try *= 0.9 * pow(tolerance / max_err, 1.0/(n_order));
	}
	else
	{
		dt_try *= 5.0;
	}
}

void integrator::update_counters(uint16_t iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;
}
