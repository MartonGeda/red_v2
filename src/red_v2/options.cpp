#include <iostream>
#include <fstream>

#include "options.h"
#include "integrator.h"
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "int_rungekutta5.h"
#include "int_rungekutta7.h"
#include "parameter.h"
#include "tbp1D.h"
#include "rtbp1D.h"
#include "tbp3D.h"
#include "rtbp3D.h"
#include "threebody.h"

#include "redutil2.h"
#include "constants.h"
#include "macro.h"

using namespace std;
using namespace redutil2;

options::options(int argc, const char** argv) :
	param(0x0)
{
	create_default();
	parse(argc, argv);
	param = new parameter(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_PARAMETER], print_to_screen);
}

options::~options() 
{
}

void options::create_default()
{
	test                = false;
	verbose             = false;
	print_to_screen     = false;
	ef                  = false;

	id_dev              = 0;
	n_change_to_cpu     = 100;

	comp_dev            = COMP_DEV_CPU;
	g_disk_model        = GAS_DISK_MODEL_NONE;

	out_fn[OUTPUT_NAME_LOG]            = "log";
	out_fn[OUTPUT_NAME_INFO]           = "info";
	out_fn[OUTPUT_NAME_EVENT]          = "event";
	out_fn[OUTPUT_NAME_SOLUTION_INFO]  = "solution_info";
	out_fn[OUTPUT_NAME_SOLUTION_DATA]  = "solution_data";
	out_fn[OUTPUT_NAME_INTEGRAL]       = "integral";
	out_fn[OUTPUT_NAME_INTEGRAL_EVENT] = "integral_event";
}

void options::parse(int argc, const char** argv)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];


		if (     p == "-m")
		{
			i++;
			string value = argv[i];
			if (     value == "tbp1D")
			{
				dyn_model = DYN_MODEL_TBP1D;
			}
			else if (value == "tbp3D")
			{
				dyn_model = DYN_MODEL_TBP3D;
			}
			else if (value == "rtbp1D")
			{
				dyn_model = DYN_MODEL_RTBP1D;
			}
			else if (value == "rtbp3D")
			{
				dyn_model = DYN_MODEL_RTBP3D;
			}
			else if (value == "threebody")
			{
				dyn_model = DYN_MODEL_THREEBODY;
			}
			else
			{
				throw string("Invalid dynamical model: " + value + ".");
			}
		}
		else if (p == "-t")
		{
			test = true;
		}
		else if (p == "-v")
		{
			verbose = true;
		}
		else if (p == "-pts")
		{
			verbose = true;
			print_to_screen = true;
		}
		else if (p == "-ef")
		{
			ef = true;
		}

		else if (p == "-id_dev")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id_dev = atoi(argv[i]);
		}
		else if (p == "-n_chg")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			n_change_to_cpu = atoi(argv[i]);
		}

		else if (p == "-cpu")
		{
			comp_dev = COMP_DEV_CPU;
		}
		else if (p == "-gpu")
		{
			comp_dev = COMP_DEV_GPU;
		}
		else if (p == "-ga")
		{
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
			g_disk_model = GAS_DISK_MODEL_ANALYTIC;
		}
		else if (p == "-gf")
		{
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
			g_disk_model = GAS_DISK_MODEL_FARGO;
		}

		else if (p == "-i")
		{
			i++;
			in_fn[INPUT_NAME_START_FILES] = argv[i];
		}
		else if (p == "-p")
		{
			i++;
			in_fn[INPUT_NAME_PARAMETER] = argv[i];
		}

		else if (p == "-idir")
		{
			i++;
			dir[DIRECTORY_NAME_IN] = argv[i];
		}
		else if (p == "-odir")
		{
			i++;
			dir[DIRECTORY_NAME_OUT] = argv[i];
		}

		else if (p == "-h")
		{
			print_usage();
			exit(EXIT_SUCCESS);
		}
		else
		{
			throw string("Invalid switch on command-line: " + p + ".");
		}
		i++;
	}

	if (0 == dir[DIRECTORY_NAME_OUT].length())
	{
		dir[DIRECTORY_NAME_OUT] = dir[DIRECTORY_NAME_IN];
	}
}

void options::get_solution_path(string& path_si, string &path_sd)
{
	if (0 < in_fn[INPUT_NAME_START_FILES].length())
	{
		string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_START_FILES]);
		ifstream file(path.c_str(), ifstream::in);
		if (file)
		{
			uint32_t n = 0;
			string str;
			while (getline(file, str))
			{
				if (0 == n)
				{
                    in_fn[INPUT_NAME_IC_INFO] = str;
				}
				if (1 == n)
				{
                    in_fn[INPUT_NAME_IC_DATA] = str;
				}
				n++;
			} 	
			file.close();
		}
		else
		{
			throw string("The file '" + path + "' could not opened.");
		}
	}
	else
	{
		throw string("The -i option was not provided.");
	}
    path_si = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_IC_INFO]);
    path_sd = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_IC_DATA]);
}

ode* options::create_tbp1D()
{
    string path_si;
    string path_sd;

    get_solution_path(path_si, path_sd);
    tbp1D* model = new tbp1D(path_si, path_sd, 1, comp_dev);
	
	// TODO: transform time variables in opt
	param->stop_time = model->t + param->simulation_length;

	return model;
}

ode* options::create_rtbp1D()
{
//    string path_si;
//    string path_sd;
//
//    get_solution_path(path_si, path_sd);
	rtbp1D* model = new rtbp1D(1, comp_dev);
//
//	string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA]);
//	model->load(path);
//	model->calc_integral();
//
//	model->tout = model->t;
//
//	// TODO: transform time variables in opt
//	param->stop_time = param->start_time + param->simulation_length;
//
	return model;
}

ode* options::create_tbp3D()
{
	tbp3D* model = new tbp3D(1, comp_dev);
//	
//	string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA]);
//	model->load(path);
//	// TODO: calc_integrals
//	model->calc_integral();
//
//	model->t    = model->h_epoch[0];
//	model->tout = model->t;
//
//	// TODO: transform time variables in opt
//	param->stop_time = param->start_time + param->simulation_length;
//
	return model;
}

ode* options::create_rtbp3D()
{
	rtbp3D* model = new rtbp3D(1, comp_dev);
//	
//	string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA]);
//	model->load(path);
//	model->calc_integral();
//
//	model->t    = model->h_epoch[0];
//	model->tout = model->t;
//
//	// TODO: transform time variables in opt
//	param->stop_time = param->start_time + param->simulation_length;
//
	return model;
}

ode* options::create_threebody() //TODO
{
	threebody* model = new threebody(1, comp_dev);
//	
//	string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA]);
//	model->load(path);
//	// TODO: calc_integrals
//	model->calc_integral();
//
//	model->t    = model->h_epoch[0];
//	model->tout = model->t;
//
//	// TODO: transform time variables in opt
//	param->stop_time = param->start_time + param->simulation_length;
//
	return model;
}

integrator* options::create_integrator(ode& f, ttt_t dt)
{
	integrator* intgr = 0x0;

	switch (param->int_type)
	{
	case INTEGRATOR_EULER:
		intgr = new euler(f, dt, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTA2:
		intgr = new int_rungekutta2(f, dt, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTA4:
		intgr = new int_rungekutta4(f, dt, param->adaptive, param->tolerance, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTAFEHLBERG56:
		intgr = new int_rungekutta5(f, dt, param->adaptive, param->tolerance, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTAFEHLBERG78:
		intgr = new int_rungekutta7(f, dt, param->adaptive, param->tolerance, comp_dev);
		break;
	default:
		throw string("Requested integrator is not implemented.");
	}

	if (param->error_check_for_tp)
	{
		intgr->error_check_for_tp = true;
	}

	return intgr;
}

void options::print_usage()
{
	cout << "Usage: red.cuda <parameterlist>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -b      | --benchmark                    : run benchmark to find out the optimal number of threads per block" << endl;
	cout << "     -t      | --test                         : run tests" << endl;
	cout << "     -v      | --verbose                      : verbose mode (log all event during the execution fo the code to the log file)" << endl;
	cout << "     -pts    | --print_to_screen              : verbose mode and print everything to the standard output stream too" << endl;
	cout << "     -ef     |                                : use extended file names (use only for debuging purposes)" << endl;

	cout << "     -i_dt   | --info-dt <number>             : the time interval in seconds between two subsequent information print to the screen (default value is 5 sec)" << endl;
	cout << "     -d_dt   | --dump-dt <number>             : the time interval in seconds between two subsequent data dump to the hard disk (default value is 3600 sec)" << endl;

	cout << "     -id_dev | --id_active_device <number>    : the id of the device which will execute the code (default value is 0)" << endl;
	cout << "     -n_chg  | --n_change_to_cpu <number>     : the threshold value for the total number of SI bodies to change to the CPU (default value is 100)" << endl;

	cout << "     -gpu    | --gpu                          : execute the code on the graphics processing unit (GPU) (default value is true)" << endl;
	cout << "     -cpu    | --cpu                          : execute the code on the cpu if required by the user or if no GPU is installed (default value is false)" << endl;

	cout << "     -log    | --log-filename <filename>      : the name of the file where the details of the execution of the code will be stored (default value is log.txt)" << endl;
	cout << "     -info   | --info-filename <filename>     : the name of the file where the runtime output of the code will be stored (default value is info.txt)" << endl;
	cout << "     -event  | --event-filename <filename>    : the name of the file where the details of each event will be stored (default value is event.txt)" << endl;
	cout << "     -result | --result-filename <filename>   : the name of the file where the simlation data for a time instance will be stored (default value is solution.txt)" << endl;

	cout << "     -iDir   | --inputDir <directory>         : the directory containing the input files"  << endl;
	cout << "     -oDir   | --outputDir <directory>        : the directory where the output files will be stored (if omitted the input directory will be used)" << endl;
	cout << "     -ic     | --initial_condition <filename> : the file containing the initial conditions"  << endl;
	cout << "     -p      | --parameter <filename>         : the file containing the parameters of the simulation"  << endl;

	cout << "     -ga     | --analytic_gas_disk <filename> : the file containing the parameters of an analyticaly prescribed gas disk"  << endl;
	cout << "     -gf     | --fargo_gas_disk <filename>    : the file containing the details of the gas disk resulted from FARGO simulations"  << endl;

	cout << "     -h      | --help                         : print this help" << endl;
}
