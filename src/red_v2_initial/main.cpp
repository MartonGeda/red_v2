#include <iostream>
#include <fstream>
#include <string>

#include "constants.h"
#include "type.h"
#include "redutil2.h"

using namespace std;
using namespace redutil2;

static string fn_info;
static string fn_data;

namespace model
{
	void print_start_files(string& dir)
	{
		string path = file::combine_path(dir, "start_files.txt");
		printf("Writing %s to disk.\n", path.c_str());

		ofstream sout(path.c_str(), ios::out);
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
	}

	namespace tbp1D
    {
        // parameter of the problem
        tbp_t::param_t p;
       	// Epoch for the initial condition
        var_t t0;
        // Initial conditions
        var_t* y;
        // Metadata of the object
        tbp_t::metadata_t md;
        // Initial stepsize for the integrator
        var_t dt0;
        
        void print(string& dir, string& filename)
        {
           	ofstream sout;

            string path = file::combine_path(dir, fn_info);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
				file::tbp::print_solution_info(sout, t0, dt0, DATA_REPRESENTATION_ASCII);
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
                file::tbp::print_solution_data(sout, 1, 1, 2, &md, (var_t*)&p, y, DATA_REPRESENTATION_ASCII);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

			print_start_files(dir);
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

			// Set the parameters of the problem
            p.mu  = constants::Gauss2 * (1.0 + 1.0);
			// Set the initial conditions at t0
            t0   = 0.0;
            y[0] = 1.0;    /* x0  */
            y[1] = 0.0;    /* vx0 */
			// Set the object metadata
            md.id = 1;
			// Set the initial stepsize for the integrator (should be modell dependent)
            dt0   = 1.0e-4;

            print(dir, filename);

            FREE_HOST_VECTOR((void **)&(y));
        }
    } /* namespace tbp1D */

	namespace tbp2D
	{
        // parameter of the problem
        tbp_t::param_t p;
       	// Epoch for the initial condition
        var_t t0;
        // Initial conditions
        var_t* y;
        // Metadata of the object
        tbp_t::metadata_t md;
        // Initial stepsize for the integrator
        var_t dt0;
        
        void print(string& dir, string& filename)
        {
           	ofstream sout;

            string path = file::combine_path(dir, fn_info);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
                file::tbp::print_solution_info(sout, t0, dt0, DATA_REPRESENTATION_ASCII);
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
                file::tbp::print_solution_data(sout, 1, 1, 4, &md, (var_t*)&p, y, DATA_REPRESENTATION_ASCII);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

			print_start_files(dir);
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
            
            ALLOCATE_HOST_VECTOR((void**)&(y), 4 * sizeof(var_t));

			// Set the parameter of the problem
            p.mu = constants::Gauss2 * (1.0 + 1.0);
			// Set the initial conditions at t0
            t0   = 0.0;
			// Set the initial orbital elements
			orbelem_t oe = {1.0, 0.1, 0.0, 0.0, 0.0, 0.0};
			var3_t r0 = {0, 0, 0};
			var3_t v0 = {0, 0, 0};
			// Calculate the initial position and velocity vectors
			tools::calc_phase(p.mu, &oe, &r0, &v0);
			y[0] = r0.x;
			y[1] = r0.y;
			y[2] = v0.x;
            y[3] = v0.y;
			// Set the object metadata
            md.id = 1;
			// Set the initial stepsize for the integrator (should be modell dependent)
            dt0   = 1.0e-4;

            print(dir, filename);

            FREE_HOST_VECTOR((void **)&(y));
        }
	} /* namespace tbp2D */

	namespace rtbp1D
    {
        // parameter of the problem
        tbp_t::param_t p;
       	// Value of the independent variable for the initial condition
        var_t s0;
        // Initial conditions
        var_t* y;
        // Metadata of the object
        tbp_t::metadata_t md;
        // Initial stepsize for the integrator
        var_t ds0;
        
        void print(string& dir, string& filename)
        {
           	ofstream sout;

            string path = file::combine_path(dir, fn_info);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
				file::tbp::print_solution_info(sout, s0, ds0, DATA_REPRESENTATION_ASCII);
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
                file::rtbp::print_solution_data(sout, 1, 1, 2, &md, (var_t*)&p, y, 1, DATA_REPRESENTATION_ASCII);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

			print_start_files(dir);
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
            
            ALLOCATE_HOST_VECTOR((void**)&(y), 3 * sizeof(var_t));

			// Set the parameters of the problem
            p.mu  = constants::Gauss2 * (1.0 + 1.0);
			// Set and compute the initial conditions at s0 (fictitious time)
            s0   = 0.0;
			var_t x0   = 1.0;
			var_t vx0  = 0.0;
			tools::rtbp1D::transform_x2u(x0, vx0, y[0], y[1]);
			var_t _x0  = 0.0;
			var_t _vx0 = 0.0;
			tools::rtbp1D::transform_u2x(y[0], y[1], _x0, _vx0);

			y[2] = 0.0;   // t0
			// Set the object metadata
            md.id = 1;
			// Set the initial stepsize for the integrator (should be model dependent)
            ds0   = 1.0e-4;

            print(dir, filename);

            FREE_HOST_VECTOR((void **)&(y));
        } /* create() */
    } /* namespace tbp1D */

	namespace rtbp2D
    {
        // parameter of the problem
        tbp_t::param_t p;
       	// Value of the independent variable for the initial condition
        var_t s0;
        // Initial conditions
        var_t* y;
        // Metadata of the object
        tbp_t::metadata_t md;
        // Initial stepsize for the integrator
        var_t ds0;
        
		var_t reg_calc_integral(var_t mu, var_t* y)
		{
			var_t r  = SQR(y[0]) + SQR(y[1]);
			var_t v2 = SQR(y[2]) + SQR(y[3]);
			var_t h = (2.0*v2 - mu) / r;

			return h;
		}

		var_t calc_integral(var_t mu, var_t* y)
		{
			var_t r  = sqrt(SQR(y[0]) + SQR(y[1]));
			var_t v2 = SQR(y[2]) + SQR(y[3]);
			var_t h = 0.5*v2 - mu/r;
			return h;
		}

		void print(string& dir, string& filename)
        {
           	ofstream sout;

            string path = file::combine_path(dir, fn_info);
	        printf("Writing %s to disk.\n", path.c_str());
       		sout.open(path.c_str(), ios::out);
            if (sout)
            {
                file::tbp::print_solution_info(sout, s0, ds0, DATA_REPRESENTATION_ASCII);
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
                file::rtbp::print_solution_data(sout, 1, 1, 4, &md, (var_t*)&p, y, 2, DATA_REPRESENTATION_ASCII);
            }
            else
            {
                throw string("Cannot open " + path + ".");
            }
            sout.close();

			print_start_files(dir);
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
            
            ALLOCATE_HOST_VECTOR((void**)&(y), 5 * sizeof(var_t));

			// Set the parameters of the problem
            p.mu = constants::Gauss2 * (1.0 + 1.0);
			// Set and compute the initial conditions at s0 (fictitious time)
            s0   = 0.0;
			// Set the initial orbital elements
			orbelem_t oe = {1.0, 0.1, 0.0, 0.0, 0.0, 0.0};
			var3_t r0 = {0, 0, 0};
			var3_t v0 = {0, 0, 0};
			// Calculate the initial position and velocity vectors
			tools::calc_phase(p.mu, &oe, &r0, &v0);

			var2_t r02D = {r0.x, r0.y};
			var2_t v02D = {v0.x, v0.y};
			var2_t u    = {0, 0};
			var2_t up   = {0, 0};
			tools::rtbp2D::transform_x2u(r02D, u);
            tools::rtbp2D::transform_xd2up(u, v02D, up);

			y[0] = u.x;
			y[1] = u.y;
			y[2] = up.x;
			y[3] = up.y;
			y[4] = 0.0;   // t0
			// Set the object metadata
            md.id = 1;
			// Set the initial stepsize for the integrator (should be model dependent)
            ds0   = 1.0e-4;

            print(dir, filename);

            FREE_HOST_VECTOR((void **)&(y));
        } /* create() */
    } /* namespace tbp2D */

	namespace nbody
	{

	} /* namespace nbody */

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

		fn_info = filename + ".info.txt";
		fn_data = filename + ".data.txt";
        //model::tbp1D::create(odir, filename);
        //model::tbp2D::create(odir, filename);
        //model::rtbp1D::create(odir, filename);
        model::rtbp2D::create(odir, filename);
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
		return (EXIT_FAILURE);
	}

	return (EXIT_SUCCESS);
}
