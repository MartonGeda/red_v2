#if 1
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand, malloc       */
#include <time.h>       /* time                      */

#include <iostream>
#include <string>

#include "constants.h"
#include "type.h"
#include "redutil2.h"

using namespace std;
using namespace redutil2;

int comp_value(var_t v1, var_t v2, var_t tol, char* lpad, char* text)
{
	int result = 0;

	var_t d = fabs(v1 - v2);
	if (tol < d)
	{
		printf("%s%s = %25.15lg\n", lpad, text, d);
		result = 1;
	}

	return result;
}

int comp_oe(orbelem_t &oe1, orbelem_t& oe2, var_t tol, char* lpad)
{
	int result = comp_value(oe1.sma, oe2.sma, tol, lpad, "Abs(Delta(sma ))");
	result += comp_value(oe1.ecc, oe2.ecc, tol, lpad, "Abs(Delta(ecc ))");
	result += comp_value(oe1.inc, oe2.inc, tol, lpad, "Abs(Delta(inc ))");
	result += comp_value(oe1.peri, oe2.peri, tol, lpad, "Abs(Delta(peri))");
	result += comp_value(oe1.node, oe2.node, tol, lpad, "Abs(Delta(node))");
	result += comp_value(oe1.mean, oe2.mean, tol, lpad, "Abs(Delta(mean))");
	return result;
}

int comp_2D_vectors(var2_t &v1, var2_t &v2, var_t tol, char* lpad)
{
	int result = comp_value(v1.x, v2.x, tol, lpad, "Abs(Delta(v1.x - v2.x))");
	result += comp_value(v1.y, v2.y, tol, lpad, "Abs(Delta(v1.y - v2.y))");
	return result;
}

var_t random(var_t x0, var_t x1)
{
	return (x0 + ((var_t)rand() / RAND_MAX) * (x1 - x0));
}

void test_calc_ephemeris()
{
	// Test calculate phase from orbital elements and vice versa
	{
		const char func_name[] = "calc_phase";
		char lpad[] = "        ";
		/*
		 * The units are:
		 *     Unit name         | Unit symbol | Quantity name
		 *     -----------------------------------------------
		 *     Astronomical unit |          AU | length
		 *     Solar mass        |           S | mass
		 *     Mean solar day    |           D | time
		 */

		srand (time(NULL));
		// parameter of the problem
		tbp_t::param_t p;
            
		// Set the parameter of the problem
		p.mu = constants::Gauss2 * (1.0 + 1.0);
		// Set the initial orbital elements
		orbelem_t oe1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		orbelem_t oe2 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};

		var_t tol = 1.0e-14;
		for (int i = 0; i < 100; i++)
		{
			oe1.sma = random(0.1, 10.0);
			oe1.ecc = random(0.0, 0.8);
			oe1.inc = random(0.0, PI);
			oe1.peri =random(0.0, TWOPI);
			oe1.node =random(0.0, TWOPI);
			oe1.mean =random(0.0, TWOPI);
			// Calculate the position and velocity vectors from orbital elements
			tools::calc_phase(p.mu, &oe1, &r0, &v0);
			// Calculate the orbital elements from position and velocity vectors
			tools::calc_oe(p.mu, &r0, &v0, &oe2);
	
			int ret_val = comp_oe(oe1, oe2, tol, lpad);
			if (0 < ret_val)
			{
				printf("    TEST '%s' failed with tolerance level: %g\n", func_name, tol);
			}
			else
			{
				printf("    TEST '%s' passed with tolerance level: %g\n", func_name, tol);
			}
		}
	} /* Test calc_phase() and calc_oe() functions */
}

void test_rtbp2d_calc_energy()
{
	// Test tools::tbp::calc_integral() and tools::rtbp2D::calc_integral() functions
	{
		const char func_name[] = "tools::tbp::calc_integral";
		char lpad[] = "        ";

	    /*
	     * The units are:
	     *     Unit name         | Unit symbol | Quantity name
	     *     -----------------------------------------------
	     *     Astronomical unit |          AU | length
	     *     Solar mass        |           S | mass
	     *     Mean solar day    |           D | time
	     */

		srand(0);

		orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};

		var_t mu = constants::Gauss2*(1.0 + 1.0);
		var_t tol = 1.0e-15;
		for (int i = 0; i < 10; i++)
		{
			// Set the initial orbital elements
			oe.sma  = random(0.1, 10.0);
			oe.ecc  = random(0.0, 0.8);
			oe.inc  = 0.0;
			oe.peri = random(0.0, TWOPI);
			oe.node = 0.0;
			oe.mean = random(0.0, TWOPI);
			// Calculate the position and velocity vectors from orbital elements
			tools::calc_phase(mu, &oe, &r0, &v0);

			// Set the starting coordinate and velocity vectors
			var2_t r  = {r0.x, r0.y};
			var2_t v  = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};
			tools::rtbp2D::transform_x2u(r, u);
			tools::rtbp2D::transform_xd2up(u, v, up);

			var_t hs = tools::tbp::calc_integral(mu, r, v);
			var_t hr = tools::rtbp2D::calc_integral(mu, u, up);

			printf("    hs = %25.15le\n", hs);
			printf("    hr = %25.15le\n", hr);
		}

		// Calculate the energy along a Kepler-orbit
		oe.sma  = 1.5;
		oe.ecc  = 0.8;
		oe.inc  = 0.0;
		oe.peri = 0.0;
		oe.node = 0.0;
		oe.mean = 0.0;
		do
		{
			tools::calc_phase(mu, &oe, &r0, &v0);
			var2_t r  = {r0.x, r0.y};
			var2_t v  = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};
			tools::rtbp2D::transform_x2u(r, u);
			tools::rtbp2D::transform_xd2up(u, v, up);

			var_t hs = tools::tbp::calc_integral(mu, r, v);
			var_t hr = tools::rtbp2D::calc_integral(mu, u, up);
			var_t d = sqrt(SQR(r.x) + SQR(r.y));
			printf("mean = %25.15le d = %25.15le hs = %25.15le hr = %25.15le\n", oe.mean * constants::RadianToDegree, d, hs, hr);

			oe.mean += 1.0 * constants::DegreeToRadian;
		} while (oe.mean <= TWOPI);
	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}

void test_rtbp2d_trans()
{
	// Test ...
	{
		const char func_name[] = "tools::rtbp2D::transform_x2u";
		char lpad[] = "        ";

	    /*
	     * The units are:
	     *     Unit name         | Unit symbol | Quantity name
	     *     -----------------------------------------------
	     *     Astronomical unit |          AU | length
	     *     Solar mass        |           S | mass
	     *     Mean solar day    |           D | time
	     */

		srand(0);

		var_t tol = 1.0e-15;
		for (int i = 0; i < 100; i++)
		{
			// Set the starting coordinate and velocity vectors
			var2_t r1 = {0.1 - random(0, 2.0), 0.1 - random(0, 2.0)};
			var2_t v1 = {-i*0.05, -i*0.025};

			var2_t r2 = {0.0, 0.0};
			var2_t v2 = {0.0, 0.0};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};
			tools::rtbp2D::transform_x2u(r1, u);
			tools::rtbp2D::transform_xd2up(u, v1, up);
			tools::rtbp2D::transform_u2x(u, r2);
			tools::rtbp2D::transform_up2xd(u, up, v2);

			int ret_val = comp_2D_vectors(r1, r2, tol, lpad); 
			if (0 < ret_val)
			{
				printf("    TEST '%s' failed with tolerance level: %g\n", func_name, tol);
			}
			else
			{
				printf("    TEST '%s' passed with tolerance level: %g\n", func_name, tol);
			}
			ret_val = comp_2D_vectors(v1, v2, tol, lpad); 
			if (0 < ret_val)
			{
				printf("    TEST '%s' failed with tolerance level: %g\n", func_name, tol);
			}
			else
			{
				printf("    TEST '%s' passed with tolerance level: %g\n", func_name, tol);
			}
		}
		// Calculate the energy
		//y[0] = r0.x, y[1] = r0.y;
		//y[2] = v0.x, y[3] = v0.y;
		//var_t h = calc_integral(p.mu, y);

		//y[0] = u.x;
		//y[1] = u.y;
		//y[2] = up.x;
		//y[3] = up.y;
		//var_t reg_h = reg_calc_integral(p.mu, y);

	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}


/*
p [-1:1][-1:1]'x2u_1.txt' u 2:3 w l, '' u 4:5 w l, 'x2u_2.txt' u 2:3 w l, '' u 4:5 w l, 'x2u_3.txt' u 2:3 w l, '' u 4:5 w l, 'x2u_4.txt' u 2:3 w l, '' u 4:5 w l
p [-1:1][-1:1]'_x2u_1.txt' u 2:3 w l, '' u 4:5 w l, '_x2u_2.txt' u 2:3 w l, '' u 4:5 w l, '_x2u_3.txt' u 2:3 w l, '' u 4:5 w l, '_x2u_4.txt' u 2:3 w l, '' u 4:5 w l

p [-1:1][-1:1]'q1.txt' u 2:3 w l, '' u 4:5 w l, 'q2.txt' u 2:3 w l, '' u 4:5 w l, 'q3.txt' u 2:3 w l, '' u 4:5 w l, 'q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'q1.txt' u 6:7 w l, '' u 8:9 w l, 'q2.txt' u 6:7 w l, '' u 8:9 w l, 'q3.txt' u 6:7 w l, '' u 8:9 w l, 'q4.txt' u 6:7 w l, '' u 8:9 w l

*/
int main()
{
	try
	{
		//test_calc_ephemeris();
		//test_rtbp2d_trans();
		test_rtbp2d_calc_energy();

        return 0;

		var_t mu = constants::Gauss2*(1.0 + 1.0);
		orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};

		var_t R = 0.5;
		oe.sma = R;
		var_t alpha = 3.0*TWOPI/4.0;
		oe.mean = alpha;
		do
		{
			var2_t x = {R * cos(alpha), R * sin(alpha)};
			var2_t u = {0.0, 0.0};
			tools::rtbp2D::transform_x2u(x, u);
			x.x = x.y = 0.0;
			tools::rtbp2D::transform_u2x(u, x);
			
			tools::calc_phase(mu, &oe, &r0, &v0);
			var2_t r  = {r0.x, r0.y};
			var2_t xd = {v0.x, v0.y};
			var2_t up = {0.0, 0.0};
			tools::rtbp2D::transform_xd2up(u, xd, up);
			xd.x = xd.y = 0.0;
			tools::rtbp2D::transform_up2xd(u, up, xd);

			printf("%22.15le %22.15le %22.15le %22.15le %22.15le %22.15le %22.15le %22.15le %22.15le\n", alpha, x.x, x.y, u.x, u.y, xd.x, xd.y, up.x, up.y);
			alpha += 0.01;
			oe.mean = alpha;
		} while (4.0*TWOPI/4.0 >= alpha);
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

}

#endif
