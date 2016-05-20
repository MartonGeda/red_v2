#include <iostream>
#include <fstream>
#include <string>

#include "constants.h"
#include "type.h"
#include "redutil2.h"

using namespace std;
using namespace redutil2;

namespace model
{
    namespace tbp1D
    {
        // parameter of the problem
        tbp1D_t::param_t p;
       	// Epoch for the initial condition
        ttt_t t0;
        // Initial conditions
        var_t* y;
        // Metadata of the object
        tbp1D_t::metadata_t md;
        // Initial stepsize for the integrator
        ttt_t dt0;
        
        void print(string& dir, string& filename)
        {
           	ofstream sout;

        	string fn_info = filename + ".info.txt";
        	string fn_data = filename + ".data.txt";

            string path = file::combine_path(dir, fn_info);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
                file::tbp1D::print_solution_info_ascii(sout, t0, dt0);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

	        path = file::combine_path(dir, fn_data);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
                file::tbp1D::print_solution_data_ascii(sout, 1, 1, 2, &md, (var_t*)&p, y);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

	        path = file::combine_path(dir, "start_files.txt");
	        printf("Writing %s to disk.\n", path.c_str());
            sout.open(path.c_str(), ios::out);
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

        } /* print() */

        void create(string& dir, string& filename)
        {
	        /*
	         * The units are:
	         *     Unit name         | Unit symbol | Quantity name
	         *     -----------------------------------------------
	         *     Astronomical unit |          AU | length
	         *     Solar mass        |           S | mass
	         *     Mean solar day    |           D | time
	         */
            
            ALLOCATE_HOST_VECTOR((void**)&(y), 2 * sizeof(var_t));

            p.mu  = constants::Gauss2 * (1.0 + 1.0);
            t0    = 0.0;
            y[0]  = 1.0;    /* x0  */
            y[1]  = 0.0;    /* vx0 */
            md.id = 1;

            dt0   = 1.0e-4;

            print(dir, filename);

            FREE_HOST_VECTOR((void **)&(y));
        }

    } /* namespace tbp1D_t */
} /* namespace model */


int parse_options(int argc, const char **argv, string &odir, string &filename)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "-odir")
		{
			i++;
			odir = argv[i];
		}
		else if (p == "-f")
		{
			i++;
			filename = argv[i];
		}
		else
		{
			throw string("Invalid switch on command-line.");
		}
		i++;
	}

	return 0;
}

int main(int argc, const char **argv)
{
	string odir;
	string filename;

	try
	{
        if (2 > argc)
        {
            throw string("Missing command line arguments.");
        }
		parse_options(argc, argv, odir, filename);

        model::tbp1D::create(odir, filename);
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
		return (EXIT_FAILURE);
	}

	return (EXIT_SUCCESS);
}
