#include <iostream>
#include <iomanip>
#include <fstream>

#include "rtbp1D.h"

#include "redutil2.h"
#include "constants.h"

using namespace std;
using namespace redutil2;


rtbp1D::rtbp1D(uint16_t n_ppo, computing_device_t comp_dev) :
	ode(1, 3, n_ppo, 1, comp_dev)
{
	initialize();
	allocate_storage();
}

rtbp1D::~rtbp1D()
{
	deallocate_storage();
}

void rtbp1D::initialize()
{
	h_md    = 0x0;
	h_epoch = 0x0;

	h       = 0.0;            // energy
	h_y[2]  = 0.0;            // s_0: fictitious time
}

void rtbp1D::allocate_storage()
{
	allocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void rtbp1D::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md),    n_obj * sizeof(tbp1D_t::metadata_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_epoch), n_obj * sizeof(var_t));
}

void rtbp1D::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md),    n_obj * sizeof(tbp1D_t::metadata_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_epoch), n_obj * sizeof(var_t));
}

void rtbp1D::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void rtbp1D::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
	FREE_HOST_VECTOR((void **)&(h_epoch));
}

void rtbp1D::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(h_md));
	FREE_DEVICE_VECTOR((void **)&(h_epoch));
}

void rtbp1D::trans_to_descartes_var(var_t& x, var_t& vx)
{
	x  = SQR(h_y[0]);
	vx = (2.0/h_y[0]) * (h_y[0] * h_y[1]);
}

void rtbp1D::calc_integral()
{
	const tbp1D_t::param_t* p = (tbp1D_t::param_t*)h_p;

	h = (2.0 * SQR(h_y[1]) - p[0].mu ) / SQR(h_y[0]);
}

void rtbp1D::calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	if (COMPUTING_DEVICE_CPU == comp_dev)
	{
		cpu_calc_dy(stage, curr_t, y_temp, dy);
	}
	else
	{
		gpu_calc_dy(stage, curr_t, y_temp, dy);
	}
}

void rtbp1D::cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	dy[0] = y_temp[1];                // dy1 / ds = y2
	dy[1] = (h / 2.0) * y_temp[0];    // dy2 / ds = h/2 * y1
	dy[2] = SQR(y_temp[0]);           // dy3 / ds = y1^2
}

void rtbp1D::gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void rtbp1D::load(string& path)
{
	ifstream input;

	cout << "Loading " << path << " ";

	data_rep_t repres = (file::get_extension(path) == "txt" ? DATA_REPRESENTATION_ASCII : DATA_REPRESENTATION_BINARY);
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
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
		input.open(path.c_str(), ios::binary);
		if (input) 
		{
			load_binary(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

	cout << " done" << endl;
}

void rtbp1D::load_ascii(ifstream& input)
{
	tbp1D_t::param_t* p = (tbp1D_t::param_t*)h_p;

	for (uint32_t i = 0; i < n_obj; i++)
	{
		load_ascii_record(input, &h_epoch[i], &h_md[i], &p[i], &h_y[i], &h_y[i+1]);
	}
}

void rtbp1D::load_ascii_record(ifstream& input, ttt_t* t, tbp1D_t::metadata_t *md, tbp1D_t::param_t* p, var_t* x, var_t* vx)
{
	string name;

	// epoch
	input >> *t;
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

void rtbp1D::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void rtbp1D::print_solution(std::string& path, data_rep_t repres)
{
	ofstream sout;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path.c_str(), ios::out);
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path.c_str(), ios::out | ios::binary);
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}

	if (sout)
	{
		switch (repres)
		{
		case DATA_REPRESENTATION_ASCII:
			print_solution_ascii(sout);
			break;
		case DATA_REPRESENTATION_BINARY:
			print_solution_binary(sout);
			break;
		default:
			throw string("Parameter 'repres' is out of range.");
		}
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	sout.close();
}

void rtbp1D::print_solution_ascii(ofstream& sout)
{
	static uint32_t int_t_w  =  8;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	var_t x  = 0.0;
	var_t vx = 0.0;
	trans_to_descartes_var(x, vx);

	for (uint32_t i = 0; i < n_obj; i++)
    {
		uint32_t orig_idx = h_md[i].id - 1;

		sout << setw(VAR_T_W) << t << SEP                       /* time of the record [day] (double)           */
			 << setw(     30) << obj_names[orig_idx] << SEP     /* name of the body         (string = 30 char) */ 
		// Print the metadata for each object
        << setw(int_t_w) << h_md[i].id << SEP;

		// Print the parameters for each object
		for (uint16_t j = 0; j < n_ppo; j++)
		{
			uint32_t param_idx = i * n_ppo + j;
			sout << setw(VAR_T_W) << h_p[param_idx] << SEP;
		}
		// Print the regularized variables for each object
		for (uint16_t j = 0; j < n_vpo; j++)
		{
			uint32_t var_idx = i * n_vpo + j;
			sout << setw(VAR_T_W) << h_y[var_idx] << SEP;
		}
		// Print the descartes non-regularized variables for each object
		sout << setw(VAR_T_W) << x << SEP
			 << setw(VAR_T_W) << vx << endl;
	}
	sout.flush();
}

void rtbp1D::print_solution_binary(ofstream& sout)
{
	throw string("The print_solution_binary() is not implemented.");
}

void rtbp1D::print_integral(string& path)
{
	ofstream sout;

	sout.open(path.c_str(), ios::out);
	if (sout)
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

		sout << setw(VAR_T_W) << t << SEP                       /* fictitious time of the record (double)           */
			 << setw(VAR_T_W) << h_y[2] << SEP                  /* real time of the record [day] double             */
			 << h << endl;                                      /* energy of the system                             */
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	sout.close();
}
