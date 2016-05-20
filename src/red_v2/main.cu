#include <ctime>
#include <iostream>
#include <fstream>

#include "integrator.h"
#include "ode.h"
#include "options.h"
#include "rtbp1D.h"
#include "tbp1D.h"
#include "rtbp3D.h"
#include "tbp3D.h"
#include "threebody.h"

#include "tools.h"

#include "type.h"
// DEBUG
#include "constants.h"
#include "redutil2.h"

using namespace std;
using namespace redutil2;

string create_prefix(const options* opt)
{
	static const char* integrator_type_short_name[] = 
	{
				"E",
				"RK2",
				"RK4",
				"RKF5",
				"RKF7"
	};

	string prefix;

	if (opt->ef)
	{
		char sep = '_';
		string config;
#ifdef _DEBUG
		config = "D";
#else
		config = "R";
#endif
		string dev = (opt->comp_dev == COMP_DEV_CPU ? "cpu" : "gpu");
		// as: adaptive step-size, fs: fix step-size
		string adapt = (opt->param->adaptive == true ? "as" : "fs");
		
		string int_name(integrator_type_short_name[opt->param->int_type]);
		prefix += config + sep + dev + sep + adapt + sep + int_name + sep;
	}

	return prefix;
}

void print_solution(uint32_t& n_print, options* opt, ode* f, integrator* intgr, ofstream& slog)
{
	static string prefix = create_prefix(opt);
	static string ext = (DATA_REPRESENTATION_ASCII == opt->param->output_data_rep ? "txt" : "dat");

    string n_print_str = redutil2::number_to_string(n_print, OUTPUT_ORDINAL_NUMBER_WIDTH, true);

    string fn_info = prefix + opt->out_fn[OUTPUT_NAME_SOLUTION_INFO] + "_" + n_print_str + "." + ext;
	string path_si = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], fn_info);

    string fn_data = prefix + opt->out_fn[OUTPUT_NAME_SOLUTION_DATA] + "_" + n_print_str + "." + ext;
	string path_sd = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], fn_data);

	f->print_solution(path_si, path_sd, opt->param->output_data_rep);
	n_print++;

	string path = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], "start_files.txt");
	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{
		sout << fn_info << endl;
		sout << fn_data << endl;
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
    sout.close();

	//f->calc_integral(false, integrals[1]);
	//f->print_integral_data(path_integral, integrals[1]);
}

void run_simulation(options* opt, ode* f, integrator* intgr, ofstream& slog)
{
	static string prefix = create_prefix(opt);
	static string ext = (DATA_REPRESENTATION_ASCII == opt->param->output_data_rep ? "txt" : "dat");

	static string path_info           = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_INFO] + ".txt");
	static string path_event          = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_EVENT] + ".txt");
    //static string path_solution_info  = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_SOLUTION_INFO] + ext);
    //static string path_solution_data  = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_SOLUTION_DATA] + ext);
	static string path_integral       = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_INTEGRAL] + ".txt");
	static string path_integral_event = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_INTEGRAL_EVENT] + ".txt");

	ttt_t ps = 0.0;
	ttt_t dt = 0.0;

	clock_t T_CPU = 0;
	clock_t dT_CPU = 0;

	time_t time_last_info = clock();
	time_t time_last_dump = clock();


	uint32_t n_print = 0;
    if (0 < opt->in_fn[INPUT_NAME_START_FILES].length() && "data" == opt->in_fn[INPUT_NAME_IC_DATA].substr(0, 4))
	{
        string str = opt->in_fn[INPUT_NAME_IC_DATA];
		size_t pos = str.find_first_of("_");
		str = str.substr(pos + 1, OUTPUT_ORDINAL_NUMBER_WIDTH);
		n_print = atoi(str.c_str());
		n_print++;
	}
    if (0 == n_print)
    {
        print_solution(n_print, opt, f, intgr, slog);
    }
	//f->print_solution(path_solution_info, path_solution_data, opt->param->output_data_rep);
	f->calc_integral();
	/* 
	 * Main cycle
	 */
	while (f->t <= opt->param->stop_time)
	{
		// make the integration step, and measure the time it takes
		clock_t T0_CPU = clock();

		dt = intgr->step();
		dT_CPU = (clock() - T0_CPU);
		T_CPU += dT_CPU;
		ps += fabs(dt);

		if (opt->param->output_interval <= fabs(ps))
		{
			ps = 0.0;
            print_solution(n_print, opt, f, intgr, slog);
			//f->print_solution(path_solution_info, path_solution_data, opt->param->output_data_rep);
			f->calc_integral();
			f->print_integral(path_integral);	
		}

		if (opt->param->info_dt < (clock() - time_last_info) / (double)CLOCKS_PER_SEC) 
		{
			time_last_info = clock();
			//print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &T_CPU, &dT_CPU);
		}
	} /* while : main cycle*/
}

int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ofstream* slog = 0x0;
	ode*         f = 0x0;
	options*   opt = 0x0;

	//matrix4_t m = {{10,8,10,7},{1,9,10,7},{8,8,4,7},{2,8,1,3}};
	//var4_t v = {4,10,1,5};
	//matrix4_t n = tools::calc_matrix_matrix_product(m,m);
	//matrix4_t l = tools::calc_matrix_transpose(n);
	//var4_t u = tools::calc_matrix_vector_product(m,v);

	var3_t qv1 = {-8.98410183670583e-09,	-9.96565986672290e-09,	0};
	var3_t qv2 = {0.790071598004156,   0.876376842255768,                   0};
	var3_t qv3 = {0.790015741877597,   0.876342937103978 ,                  0};
	var3_t pv1 = {1.35584617114928e-10,	-1.18154635090028e-10,	0};
	var3_t pv2 = {-0.012137023259470 * 5.685826099573812e-09,   0.010261361613838 * 5.685826099573812e-09, 0};
	var3_t pv3 = {-0.011709048488151 * 5.685826099573812e-09,   0.010519195691438 * 5.685826099573812e-09, 0};

	// DEBUG
	pv1.x /= constants::Gauss, pv1.y /= constants::Gauss, pv1.z /= constants::Gauss;
	pv2.x /= constants::Gauss, pv2.y /= constants::Gauss, pv2.z /= constants::Gauss;
	pv3.x /= constants::Gauss, pv3.y /= constants::Gauss, pv3.z /= constants::Gauss;

	var4_t Q1,Q2,P1,P2;
	
	tools::trans_to_threebody(qv1,pv1,qv2,pv2,qv3,pv3,Q1,P1,Q2,P2);
	tools::trans_to_descartes(1,5.685826099573812e-09,5.685826099573812e-09,qv1,pv1,qv2,pv2,qv3,pv3,Q1,P1,Q2,P2);

	try
	{
		opt = new options(argc, argv);
		string prefix = create_prefix(opt);
		string path_log = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_LOG]) + ".txt";
		slog = new ofstream(path_log.c_str(), ios::out | ios::app);
		if (!slog)
		{
			throw string("Cannot open " + path_log + ".");
		}
		file::log_start(*slog, argc, argv, env, opt->param->get_data(), opt->print_to_screen);

		switch (opt->dyn_model)
		{
		case DYN_MODEL_TBP1D:
			f = opt->create_tbp1D();
			break;
		case DYN_MODEL_RTBP1D:
			f = opt->create_rtbp1D();
			break;
		case DYN_MODEL_TBP3D:
			f = opt->create_tbp3D();
			break;
		case DYN_MODEL_RTBP3D:
			f = opt->create_rtbp3D();
			break;
		case DYN_MODEL_THREEBODY:
			f = opt->create_threebody();
			break;
		default:
			throw string("Invalid dynamical model.");
		}

		// TODO!!!!!
		//
		ttt_t dt = min(3.0e-2, opt->param->output_interval);
		//
		// TODO!!!!!

		integrator *intgr = opt->create_integrator(*f, dt);
		// TODO: For every model it should be provieded a method to determine the minimum stepsize
		// OR use the solution provided by the Numerical Recepies
		intgr->set_dt_min(1.0e-20); // day
		intgr->set_max_iter(10);

		run_simulation(opt, f, intgr, *slog);

	} /* try */
	catch (const string& msg)
	{
		if (0x0 != slog)
		{
			file::log_message(*slog, "Error: " + msg, false);
		}
		cerr << "\nError: " << msg << endl;
	}

	if (0x0 != slog)
	{
		file::log_message(*slog, "Total time: " + tools::convert_time_t(time(NULL) - start) + " s", false);
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return (EXIT_SUCCESS);
}
