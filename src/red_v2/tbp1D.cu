#include <iostream>
#include <iomanip>
#include <fstream>

#include "tbp1D.h"

#include "redutil2.h"
#include "constants.h"

using namespace std;
using namespace redutil2;


tbp1D::tbp1D(uint16_t n_ppo, comp_dev_t comp_dev) :
	ode(1, 1, 2, n_ppo, comp_dev)
{
	initialize();
	allocate_storage();
}

tbp1D::~tbp1D()
{
	deallocate_storage();
}

void tbp1D::initialize()
{
	h_md       = 0x0;
	integral.h = 0.0;        // energy
}

void tbp1D::allocate_storage()
{
	allocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void tbp1D::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md), n_obj * sizeof(tbp1D_t::metadata_t));
}

void tbp1D::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md), n_obj * sizeof(tbp1D_t::metadata_t));
}

void tbp1D::deallocate_storage()
{
	deallocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
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

void tbp1D::calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	if (COMP_DEV_CPU == comp_dev)
	{
		cpu_calc_dy(stage, curr_t, y_temp, dy);
	}
	else
	{
		gpu_calc_dy(stage, curr_t, y_temp, dy);
	}
}

void tbp1D::calc_integral()
{
	const tbp1D_t::param_t* p = (tbp1D_t::param_t*)h_p;

	integral.h = 0.5 * SQR(h_y[1]) - p[0].mu / h_y[0];
}

void tbp1D::cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	const tbp1D_t::param_t* p = (tbp1D_t::param_t*)h_p;

	dy[0] = y_temp[1];                    // dx1 / dt = x2
	dy[1] = -p[0].mu / SQR(y_temp[0]);    // dx2 / dt = -mu / (x1*x1)
}

void tbp1D::gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void tbp1D::load(string& path)
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
	tbp1D_t::param_t* p = (tbp1D_t::param_t*)h_p;

	for (uint32_t i = 0; i < n_obj; i++)
	{
		load_ascii_record(input, &t, &h_md[i], &p[i], &h_y[i], &h_y[i+1]);
	}
}

void tbp1D::load_ascii_record(ifstream& input, ttt_t* _t, tbp1D_t::metadata_t *md, tbp1D_t::param_t* p, var_t* x, var_t* vx)
{
	string name;

	// epoch
	input >> *_t;
	// name
	input >> name;
	if (name.length() > 30)
	{
		name = name.substr(0, 30);
	}
	obj_names.push_back(name);
	// id
	input >> md->id;
	// mu = k^2*(m1 + m2)
	input >> p->mu;
	// position
	input >> *x;
	// velocity
	input >> *vx;
}

void tbp1D::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void tbp1D::print_solution(std::string& path, data_rep_t repres)
{
	ofstream sout;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path.c_str(), ios::out | ios::app);
		if (sout)
		{
			print_solution_ascii(sout);
		}
		else
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path.c_str(), ios::out | ios::app | ios::binary);
		if (sout)
		{
			print_solution_binary(sout);
		}
		else
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}
	sout.close();
}

void tbp1D::print_solution_ascii(ofstream& sout)
{
	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	for (uint32_t i = 0; i < n_obj; i++)
    {
		uint32_t orig_idx = h_md[i].id - 1;

		sout << setw(VAR_T_W) << t << SEP                       /* time of the record [day] (double)           */
			 << setw(     30) << obj_names[orig_idx] << SEP     /* name of the body         (string = 30 char) */ 
		// Print the metadata for each object
        << setw(INT_T_W) << h_md[i].id << SEP;

		// Print the parameters for each object
		for (uint16_t j = 0; j < n_ppo; j++)
		{
			uint32_t param_idx = i * n_ppo + j;
			sout << setw(VAR_T_W) << h_p[param_idx] << SEP;
		}
		// Print the variables for each object
		for (uint16_t j = 0; j < n_vpo; j++)
		{
			uint32_t var_idx = i * n_vpo + j;
			sout << setw(VAR_T_W) << h_y[var_idx];
			if (j < n_vpo - 1)
			{
				sout << SEP;
			}
			else
			{
				sout << endl;
			}
		}
	}
	sout.flush();
}

void tbp1D::print_solution_binary(ofstream& sout)
{
	throw string("The print_result_binary() is not implemented.");
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

