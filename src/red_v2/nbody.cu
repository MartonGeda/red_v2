#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include "nbody.h"

#include "redutil2.h"
#include "constants.h"

using namespace std;
using namespace redutil2;


nbody::nbody(string& path_si, string& path_sd, uint32_t n_obj, uint16_t n_ppo, comp_dev_t comp_dev) :
	ode(3, n_obj, 6, n_ppo, comp_dev)
{
	initialize();
	allocate_storage();

    load_solution_info(path_si);
    load_solution_data(path_sd);

	calc_integral();
	tout = t;
}

nbody::~nbody()
{
	deallocate_storage();
}

void nbody::initialize()
{
	h_md       = 0x0;
	d_md       = 0x0;
	md         = 0x0;
}

void nbody::allocate_storage()
{
	allocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void nbody::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md), n_obj * sizeof(nbp_t::metadata_t));
}

void nbody::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md), n_obj * sizeof(nbp_t::metadata_t));
}

void nbody::deallocate_storage()
{
	deallocate_host_storage();
	if (COMP_DEV_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void nbody::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
}

void nbody::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(d_md));
}

void nbody::calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
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

void nbody::calc_integral()
{
	static bool first_call = true;
	nbp_t::param_t* p = (nbp_t::param_t*)h_p;

	var3_t* r = (var3_t*)h_y;
	var3_t* v = (var3_t*)(h_y + 3*n_obj);

	integral.R = tools::nbp::calc_position_of_bc(n_obj, p, r);
	integral.V = tools::nbp::calc_velocity_of_bc(n_obj, p, v);
	integral.c = tools::nbp::calc_angular_momentum(n_obj, p, r, v);
	integral.h = tools::nbp::calc_total_energy(n_obj, p, r, v);

	if (first_call)
	{
		first_call = false;
		integral.R0 = integral.R;
		integral.V0 = integral.V;
		integral.c0 = integral.c;
		integral.h0 = integral.h;
	}
}

void nbody::cpu_calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	memcpy(dy, y_temp + 3*n_obj, 3*n_obj*sizeof(var_t));

	nbp_t::param_t* p = (nbp_t::param_t*)h_p;
	uint32_t offset = 3*n_obj;
	for (uint32_t i = 0; i < n_obj; i++)
	{
		var3_t r_ij = {0, 0, 0};
		for (uint32_t j = i+1; j < n_obj; j++)
		{
			r_ij.x = y_temp[j+0] - y_temp[i+0];
			r_ij.y = y_temp[j+1] - y_temp[i+1];
			r_ij.z = y_temp[j+2] - y_temp[i+2];

			var_t d2 = SQR(r_ij.x) + SQR(r_ij.y) + SQR(r_ij.z);
			var_t d = sqrt(d2);
			var_t d_3 = 1.0 / (d*d2);

			dy[offset + i + 0] += p[j].mass * d_3 * r_ij.x;
			dy[offset + i + 1] += p[j].mass * d_3 * r_ij.y;
			dy[offset + i + 2] += p[j].mass * d_3 * r_ij.z;

			dy[offset + j + 0] -= p[i].mass * d_3 * r_ij.x;
			dy[offset + j + 1] -= p[i].mass * d_3 * r_ij.y;
			dy[offset + j + 2] -= p[i].mass * d_3 * r_ij.z;
		}
	}
}

void nbody::gpu_calc_dy(uint16_t stage, var_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void nbody::load_solution_info(string& path)
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
			input >> t >> dt >> n_obj;
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
    		input.read((char*)&t,     sizeof(var_t));
	        input.read((char*)&dt,    sizeof(var_t));
			input.read((char*)&n_obj, sizeof(uint32_t));
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

void nbody::load_solution_data(string& path)
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

void nbody::load_ascii(ifstream& input)
{
	for (uint32_t i = 0; i < n_obj; i++)
	{
		// id
		input >> h_md[i].id;
		// mass 
		input >> h_p[i];
		uint32_t offset = 3*i;
		// position
		input >> h_y[offset+0] >> h_y[offset+1] >> h_y[offset+2];
		offset += 3*n_obj;
		// velocity
		input >> h_y[offset+0] >> h_y[offset+1] >> h_y[offset+2];
	}
}

void nbody::load_binary(ifstream& input)
{
	for (uint32_t i = 0; i < n_obj; i++)
	{
		input.read((char*)(h_md + i), sizeof(uint32_t));
		input.read((char*)(h_p + i), sizeof(var_t));
		var_t* y = h_p + 6*i;
		input.read((char*)y, 6*sizeof(var_t));
	}
}

void nbody::print_solution(std::string& path_si, std::string& path_sd, data_rep_t repres)
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
	file::nbp::print_solution_info(sout, t, dt, n_obj, repres);
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
	file::nbp::print_solution_data(sout, n_obj, n_ppo, n_vpo, h_md, h_p, h_y, repres);
	sout.close();
}

void nbody::print_integral(string& path)
{
	ofstream sout;

	sout.open(path.c_str(), ios::out | ios::app);
	if (sout)
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

	    sout << setw(VAR_T_W) << t << SEP                    /* time of the record [day] (double)           */
			 << setw(VAR_T_W) << integral.R.x << SEP         /* x-position of the barycenter                */
			 << setw(VAR_T_W) << integral.R.y << SEP         /* y-position of the barycenter                */
			 << setw(VAR_T_W) << integral.R.z << SEP         /* z-position of the barycenter                */
			 << setw(VAR_T_W) << integral.V.x << SEP         /* x-velocity of the barycenter                */
			 << setw(VAR_T_W) << integral.V.y << SEP         /* y-velocity of the barycenter                */
			 << setw(VAR_T_W) << integral.V.z << SEP         /* z-velocity of the barycenter                */
			 << setw(VAR_T_W) << integral.c.x << SEP         /* x-angular momentum                          */
			 << setw(VAR_T_W) << integral.c.y << SEP         /* y-angular momentum                          */
			 << setw(VAR_T_W) << integral.c.z << SEP         /* z-angular momentum                          */
			 << setw(VAR_T_W) << integral.h << endl;         /* energy of the system                        */
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	sout.close();
}
