#include <iostream>
#include <iomanip>
#include <fstream>

#include "tbp1D.h"

#include "redutil2.h"
#include "constants.h"

using namespace std;
using namespace redutil2;


tbp1D::tbp1D(string& path_si, string& path_sd, uint16_t n_ppo, comp_dev_t comp_dev) :
	ode(1, 1, 2, n_ppo, comp_dev)
{
	name = "Singular 1D two-body problem";

	initialize();
	allocate_storage();

    load_solution_info(path_si);
    load_solution_data(path_sd);

    calc_integral();
    tout = t;
}

tbp1D::~tbp1D()
{
	deallocate_storage();
}

void tbp1D::initialize()
{
	h_md       = NULL;
	d_md       = NULL;
	md         = NULL;
}

void tbp1D::allocate_storage()
{
	allocate_host_storage();
	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		allocate_device_storage();
	}
}

void tbp1D::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md), n_obj * sizeof(tbp_t::metadata_t));
}

void tbp1D::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md), n_obj * sizeof(tbp_t::metadata_t));
}

void tbp1D::deallocate_storage()
{
	deallocate_host_storage();
	if (PROC_UNIT_GPU == comp_dev.proc_unit)
	{
		deallocate_device_storage();
	}
}

void tbp1D::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
}

void tbp1D::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(d_md));
}

void tbp1D::calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	if (PROC_UNIT_CPU == comp_dev.proc_unit)
	{
		cpu_calc_dy(stage, curr_t, y_temp, dy);
	}
	else
	{
		gpu_calc_dy(stage, curr_t, y_temp, dy);
	}
}

void tbp1D::copy_metadata(copy_direction_t dir)
{
	switch (dir)
	{
	case COPY_DIRECTION_TO_DEVICE:
		copy_vector_to_device(d_md, h_md, n_obj*sizeof(tbp_t::metadata_t));
		break;
	case COPY_DIRECTION_TO_HOST:
		copy_vector_to_host(h_md, d_md, n_obj*sizeof(tbp_t::metadata_t));
	default:
		throw std::string("Parameter 'dir' is out of range.");
	}
}

void tbp1D::calc_integral()
{
	static bool first_call = true;
	const tbp_t::param_t* p = (tbp_t::param_t*)h_p;

	integral.h = 0.5 * SQR(h_y[1]) - p[0].mu / h_y[0];

	if (first_call)
	{
		integral.h0 = integral.h;
		first_call = false;
	}
}

void tbp1D::cpu_calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	const tbp_t::param_t* p = (tbp_t::param_t*)h_p;

	dy[0] = y_temp[1];                    // dx1 / dt = x2
	dy[1] = -p[0].mu / SQR(y_temp[0]);    // dx2 / dt = -mu / (x1*x1)
}

void tbp1D::gpu_calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void tbp1D::load_solution_info(string& path)
{
	ifstream input;

	cout << "Loading " << path << " ";

	data_rep_t repres = file::get_data_repres(path);
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str(), ios::in);
		if (input) 
		{
			input >> t >> dt;
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
    		input.read((char*)&t, sizeof(var_t));
	        input.read((char*)&dt, sizeof(var_t));
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}
	input.close();

	cout << " done" << endl;
}

void tbp1D::load_solution_data(string& path)
{
	ifstream input;

	cout << "Loading " << path << " ";

	data_rep_t repres = file::get_data_repres(path);
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str(), ios::in);
		if (input) 
		{
			load_ascii(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
			load_binary(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}
	input.close();

	cout << " done" << endl;
}

void tbp1D::load_ascii(ifstream& input)
{
	// id
	input >> h_md[0].id;
	// mu = k^2*(m1 + m2)
	input >> h_p[0];
	// position
	input >> h_y[0];
	// velocity
	input >> h_y[1];
}

void tbp1D::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void tbp1D::print_solution(std::string& path_si, std::string& path_sd, data_rep_t repres)
{
	ofstream sout;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path_si.c_str(), ios::out | ios::app);
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path_si.c_str(), ios::out | ios::app | ios::binary);
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}
	if (!sout)
	{
		throw string("Cannot open " + path_si + ".");
	}
	file::tbp::print_solution_info(sout, t, dt, repres);
	sout.close();

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path_sd.c_str(), ios::out | ios::app);
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path_sd.c_str(), ios::out | ios::app | ios::binary);
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}
	if (!sout)
	{
		throw string("Cannot open " + path_sd + ".");
	}
	file::tbp::print_solution_data(sout, n_obj, n_ppo, n_vpo, h_md, h_p, h_y, repres);
	sout.close();
}

void tbp1D::print_integral(string& path)
{
	ofstream sout;

	sout.open(path.c_str(), ios::out | ios::app);
	if (sout)
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

	    sout << setw(VAR_T_W) << t << SEP             /* time of the record [day] (double)           */
		     << setw(VAR_T_W) << integral.h << endl;  /* energy of the system                        */
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	sout.close();
}
