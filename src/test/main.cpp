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
			printf("%25.15le %25.15le %25.15le\n", oe.mean, hs, hr);

			oe.mean += 1.0 * constants::DegreeToRadian;
		} while (oe.mean <= TWOPI);
	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}

void test_rtbp2d_transform()
{
	// Test square (section lines)
	{
		var_t d = 0.01;
		// Q4 -> Q1
		var2_t x = {0.5, -0.5};
		var2_t u  = {0, 0};
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.y += d;
		} while (0.5 >= x.y);
		// Q1 -> Q2
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.x -= d;
		} while (-0.5 <= x.x);
		// Q2 -> Q3
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.y -= d;
		} while (-0.5 <= x.y);
		// Q3 -> Q4
		do
		{
			tools::rtbp2D::transform_x2u(x, u);
			printf("%23.15le %23.15le %23.15le %23.15le\n", x.x, x.y, u.x, u.y);
			x.x += d;
		} while (0.5 >= x.x);
	}

	return;

	// Test ellipse
	{
		const char func_name[] = "tools::rtbp2D::transform___";
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

		const var_t mu = constants::Gauss2*(1.0 + 1.0);
		orbelem_t oe = {0.5, 0.8, 0.0, 0.0, 0.0, 0.0};
		var3_t r0 = {0, 0, 0};
		var3_t v0 = {0, 0, 0};
		int i = 0;
		do
		{
			oe.mean = i * constants::DegreeToRadian;
			tools::calc_phase(mu, &oe, &r0, &v0);
			var2_t x  = {r0.x, r0.y};
			var2_t xd = {v0.x, v0.y};
			var2_t u  = {0, 0};
			var2_t up = {0, 0};

			tools::rtbp2D::transform_x2u(x, u);
			tools::rtbp2D::transform_xd2up(u, xd, up);
			x.x  = x.y  = 0.0;
			xd.x = xd.y = 0.0;

			tools::rtbp2D::transform_u2x(u, x);
			tools::rtbp2D::transform_up2xd(u, up, xd);
			// Compare the original position and velocitiy vectors with the calculated ones
			{
				var_t tol = 1.0e-15;
				var2_t x0  = {r0.x, r0.y};
				var2_t x0d = {v0.x, v0.y};
				comp_2D_vectors(x0, x, tol, lpad);
				comp_2D_vectors(x0d, xd, tol, lpad);
			}

			printf("%23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le %23.15le\n", oe.mean, x.x, x.y, u.x, u.y, xd.x, xd.y, up.x, up.y);
			if (0 < i && 0 == (i+1) % 90)
			{
				printf("\n");
			}
			i++;
		} while (360 > i);
	} /* Test tools::rtbp2D::transform_x2u() and tools::rtbp2D::transform_u2x() functions */
}


/*

cd 'C:\Work\red.cuda.Results\v2.0\Test_Copy\rtbp2D\Test_transform
a=1.0
p [-1:1][-1:1]'e_0.0_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.0_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.0_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.0_q4.txt' u 6:7 w l, '' u 8:9 w l

a=1.0
p [-1:1][-1:1]'e_0.2_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.2_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.2_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.2_q4.txt' u 6:7 w l, '' u 8:9 w l

a=1.0
p [-1:1][-1:1]'e_0.8_q1.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q2.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q3.txt' u 2:3 w l, '' u 4:5 w l, 'e_0.8_q4.txt' u 2:3 w l, '' u 4:5 w l
a=0.05
p [-a:a][-a:a]'e_0.8_q1.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q2.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q3.txt' u 6:7 w l, '' u 8:9 w l, 'e_0.8_q4.txt' u 6:7 w l, '' u 8:9 w l
*/
int main()
{
	try
	{
		//test_calc_ephemeris();
		//test_rtbp2d_trans();
		//test_rtbp2d_transform();
		test_rtbp2d_calc_energy();

	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

    return 0;
}

#endif
