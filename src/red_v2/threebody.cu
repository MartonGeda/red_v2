#include <iostream>
#include <iomanip>
#include <fstream>

#include "threebody.h"

#include "redutil2.h"
#include "constants.h"

#include "tools.h"

using namespace std;
using namespace redutil2;


threebody::threebody(uint16_t n_ppo, computing_device_t comp_dev) :
	ode(3, 8, n_ppo, 3, comp_dev)
{
	initialize();
	allocate_storage();
}

threebody::~threebody()
{
	deallocate_storage();
}

void threebody::initialize()
{
	h_md    = 0x0;
	h_epoch = 0x0;

	h       = 0.0;
}

void threebody::allocate_storage()
{
	allocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void threebody::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md),    n_obj * sizeof(threebody_t::metadata_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_epoch), n_obj * sizeof(var_t));
}

void threebody::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md),    n_obj * sizeof(threebody_t::metadata_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_epoch), n_obj * sizeof(var_t));
}

void threebody::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void threebody::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
	FREE_HOST_VECTOR((void **)&(h_epoch));
}

void threebody::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(h_md));
	FREE_DEVICE_VECTOR((void **)&(h_epoch));
}
/*
void threebody::trans_to_descartes(var3_t& qv1, var3_t& pv1, var3_t& qv2, var3_t& pv2, var3_t& qv3, var3_t& pv3, const var4_t& Q1, const var4_t& P1, const var4_t& Q2, const var4_t& P2)
{
	const threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	matrix4_t A1 = {{2*Q1.x, -2*Q1.y, -2*Q1.z, 2*Q1.w},{2*Q1.y, 2*Q1.x, -2*Q1.w, -2*Q1.z},{2*Q1.z, 2*Q1.w, 2*Q1.x, 2*Q1.y},{0,0,0,0}};
	matrix4_t A2 = {{2*Q2.x, -2*Q2.y, -2*Q2.z, 2*Q2.w},{2*Q2.y, 2*Q2.x, -2*Q2.w, -2*Q2.z},{2*Q2.z, 2*Q2.w, 2*Q2.x, 2*Q2.y},{0,0,0,0}};

	var_t R1 = SQR(Q1.x) + SQR(Q1.y) + SQR(Q1.z) + SQR(Q1.w);
	var_t R2 = SQR(Q2.x) + SQR(Q2.y) + SQR(Q2.z) + SQR(Q2.w);

	var4_t q1 = tools::calc_matrix_vector_product(tools::calc_matrix_transpose(A1),Q1);
	q1.x /= 2;
	q1.y /= 2;
	q1.z /= 2;
	
	var4_t q2 = tools::calc_matrix_vector_product(tools::calc_matrix_transpose(A2),Q2);
	q2.x /= 2;
	q2.y /= 2;
	q2.z /= 2;

	var4_t p1 = tools::calc_matrix_vector_product(tools::calc_matrix_transpose(A1),P1);
	p1.x /= 4*R1;
	p1.y /= 4*R1;
	p1.z /= 4*R1;

	var4_t p2 = tools::calc_matrix_vector_product(tools::calc_matrix_transpose(A2),P2);
	p2.x /= 4*R2;
	p2.y /= 4*R2;
	p2.z /= 4*R2;

	// q3' , q1', q2'
	qv3.x = -(p[0].m * q1.x + p[1].m * q2.x) / (p[0].m + p[1].m + p[2].m);
	qv3.y = -(p[0].m * q1.y + p[1].m * q2.y) / (p[0].m + p[1].m + p[2].m);
	qv3.z = -(p[0].m * q1.z + p[1].m * q2.z) / (p[0].m + p[1].m + p[2].m);

	qv1.x = qv3.x + q1.x;
	qv1.y = qv3.y + q1.y;
	qv1.z = qv3.z + q1.z;

	qv2.x = qv3.x + q2.x;
	qv2.y = qv3.y + q2.y;
	qv2.z = qv3.z + q2.z;

	// p1', p2', p3'
	pv1.x = p1.x;
	pv1.y = p1.y;
	pv1.z = p1.z;

	pv2.x = p2.x;
	pv2.y = p2.y;
	pv2.z = p2.z;

	pv3.x = -(p1.x + p2.x);
	pv3.y = -(p1.y + p2.y);
	pv3.z = -(p1.z + p2.z);

}

void threebody::trans_to_threebody(const var3_t& qv1, const var3_t& pv1, const var3_t& qv2, const var3_t& pv2, const var3_t& qv3, const var3_t& pv3, var4_t& Q1, var4_t& P1, var4_t& Q2, var4_t& P2)
{
	var3_t q1, q2;
	q1.x = qv1.x - qv3.x;
	q1.y = qv1.y - qv3.y;
	q1.z = qv1.z - qv3.z;
	
	q2.x = qv2.x - qv3.x;
	q2.y = qv2.y - qv3.y;
	q2.z = qv2.z - qv3.z;

	var4_t p1 = {pv1.x, pv1.y, pv1.z, 0};
	var4_t p2 = {pv2.x, pv2.y, pv2.z, 0};

	if (q1.x >= 0) {
		Q1.x = sqrt(0.5 * (tools::norm(&q1)  + q1.x));
		Q1.y = 0.5 * q1.y / Q1.x;
		Q1.z = 0.5 * q1.z / Q1.x;
		Q1.w = 0;
	}
	else {
		Q1.y = sqrt(0.5 * (tools::norm(&q1)  - q1.x));
		Q1.x = 0.5 * q1.y / Q1.y;
		Q1.z = 0;
		Q1.w = 0.5 * q1.z / Q1.y;	
	}

	if (q2.x >= 0) {
		Q2.x = sqrt(0.5 * (tools::norm(&q2)  + q2.x));
		Q2.y = 0.5 * q2.y / Q2.x;
		Q2.z = 0.5 * q2.z / Q2.x;
		Q2.w = 0;
	}
	else {
		Q2.y = sqrt(0.5 * (tools::norm(&q2)  - q2.x));
		Q2.x = 0.5 * q2.y / Q2.y;
		Q2.z = 0;
		Q2.w = 0.5 * q2.z / Q2.y;	
	}

	matrix4_t A1 = {{2*Q1.x, -2*Q1.y, -2*Q1.z, 2*Q1.w},{2*Q1.y, 2*Q1.x, -2*Q1.w, -2*Q1.z},{2*Q1.z, 2*Q1.w, 2*Q1.x, 2*Q1.y},{0,0,0,0}};
	matrix4_t A2 = {{2*Q2.x, -2*Q2.y, -2*Q2.z, 2*Q2.w},{2*Q2.y, 2*Q2.x, -2*Q2.w, -2*Q2.z},{2*Q2.z, 2*Q2.w, 2*Q2.x, 2*Q2.y},{0,0,0,0}};

	P1 = tools::calc_matrix_vector_product(A1,p1);
	P2 = tools::calc_matrix_vector_product(A2,p2);	
}
*/
void threebody::calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
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

void threebody::calc_integral() // check!!!
{
	const threebody_t::param_t* p = (threebody_t::param_t*)h_p;
	var3_t q1, q2, q3, p1, p2, p3;
	const var4_t Q1 = {h_y[0], h_y[1], h_y[2], h_y[3]};
	const var4_t Q2 = {h_y[4], h_y[5], h_y[6], h_y[7]};
	const var4_t P1 = {h_y[8], h_y[9], h_y[10], h_y[11]};
	const var4_t P2 = {h_y[12], h_y[13], h_y[14], h_y[15]};

	tools::trans_to_descartes(p[0].m, p[1].m, p[2].m, q1, p1, q2, p2, q3, p3, Q1, P1, Q2, P2);

	var_t R = sqrt(SQR(q1.x - q2.x) + SQR(q1.y - q2.y) + SQR(q1.z - q2.z));
	var_t R1 = sqrt(SQR(q1.x - q3.x) + SQR(q1.y - q3.y) + SQR(q1.z - q3.z));
	var_t R2 = sqrt(SQR(q2.x - q3.x) + SQR(q2.y - q3.y) + SQR(q2.z - q3.z));

	var_t k1 = (SQR(p1.x) + SQR(p1.y) + SQR(p1.z)) / (2 * p[0].m);
	var_t k2 = (SQR(p2.x) + SQR(p2.y) + SQR(p2.z)) / (2 * p[1].m);
	var_t k3 = (SQR(p3.x) + SQR(p3.y) + SQR(p3.z)) / (2 * p[2].m);

	var_t u = -((p[0].m * p[2].m) / R1 + (p[1].m * p[2].m) / R2 + (p[0].m * p[1].m) / R);

	h = k1+k2+k3 + u;
}

void threebody::cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	const threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	/*
	Q1 = y_temp[0], Q2 = y_temp[1], Q3 = y_temp[2], Q4 = y_temp[3],
	Q5 = y_temp[4], Q6 = y_temp[5], Q7 = y_temp[6], Q8 = y_temp[7],
	P1 = y_temp[8], P2 = y_temp[9], P3 = y_temp[10], P4 = y_temp[11],
	P5 = y_temp[12], P6 = y_temp[13], P7 = y_temp[14], P8 = y_temp[15]
	*/

	var_t R1 = SQR(y_temp[0]) + SQR(y_temp[1]) + SQR(y_temp[2]) + SQR(y_temp[3]);
	var_t R2 = SQR(y_temp[4]) + SQR(y_temp[5]) + SQR(y_temp[6]) + SQR(y_temp[7]);
	var_t R = SQR(y_temp[0] - y_temp[4]) + SQR(y_temp[1] - y_temp[5]) + SQR(y_temp[2] - y_temp[6]) + SQR(y_temp[3] - y_temp[7]);
	var_t mu13 =  (p[0].m * p[2].m) / (p[0].m + p[2].m);
	var_t mu23 =  (p[1].m * p[2].m) / (p[1].m + p[2].m);
	threebody::calc_integral(); // check if its needed or not!!
	var_t P9 = h; // check!!!!!

	// create KS matrices
	matrix4_t Q1 = {{y_temp[0],  -y_temp[1],  -y_temp[2],  y_temp[3]}, {y_temp[1],  y_temp[0],  -y_temp[3],  -y_temp[2]}, {y_temp[2],  y_temp[3],  y_temp[0],  y_temp[1]}, {0.0, 0.0, 0.0, 0.0}};	
	matrix4_t Q5 = {{y_temp[4],  -y_temp[5],  -y_temp[6],  y_temp[7]}, {y_temp[5],  y_temp[4],  -y_temp[7],  -y_temp[6]}, {y_temp[6],  y_temp[7],  y_temp[4],  y_temp[5]}, {0.0, 0.0, 0.0, 0.0}};
	matrix4_t P1 = {{y_temp[8],  -y_temp[9],  -y_temp[10], y_temp[11]},{y_temp[9],  y_temp[8],  -y_temp[11], -y_temp[10]},{y_temp[10], y_temp[11], y_temp[8],  y_temp[9]}, {0.0, 0.0, 0.0, 0.0}};
	matrix4_t P5 = {{y_temp[12], -y_temp[13], -y_temp[14], y_temp[14]},{y_temp[13], y_temp[12], -y_temp[15], -y_temp[14]},{y_temp[14], y_temp[15], y_temp[12], y_temp[13]},{0.0, 0.0, 0.0, 0.0}};

	var4_t q1 = {y_temp[0], y_temp[1], y_temp[2], y_temp[3]};
	var4_t q5 = {y_temp[4], y_temp[5], y_temp[6], y_temp[7]};
	var4_t p1 = {y_temp[8], y_temp[9], y_temp[10], y_temp[11]};
	var4_t p5 = {y_temp[12], y_temp[13], y_temp[14], y_temp[15]};
	var_t sqr_p1 = SQR(y_temp[8]) + SQR(y_temp[9]) + SQR(y_temp[10]) + SQR(y_temp[11]);
	var_t sqr_p5 = SQR(y_temp[12]) + SQR(y_temp[13]) + SQR(y_temp[14]) + SQR(y_temp[15]);

	var_t c1 = 1/(4*p[2].m);
	var_t c2 = R2/(4*mu13);
	var_t c3 = R1/(4*mu23);

	var4_t c4 = tools::calc_matrix_vector_product(tools::calc_matrix_matrix_product(Q1,tools::calc_matrix_transpose(Q5)), p5);
	var4_t c5 = tools::calc_matrix_vector_product(tools::calc_matrix_matrix_product(Q5,tools::calc_matrix_transpose(Q1)), p1);
	var4_t c6 = tools::calc_matrix_vector_product(tools::calc_matrix_matrix_product(P1,tools::calc_matrix_transpose(Q5)), p5);
	var4_t c7 = tools::calc_matrix_vector_product(tools::calc_matrix_matrix_product(P5,tools::calc_matrix_transpose(Q1)), p1);

	var_t c8 = 2*R2*P9 - 2*p[1].m*p[2].m + sqr_p5/(4*mu23) - 2*p[0].m*p[1].m*R2/R;
	var_t c9 = 2*R1*P9 - 2*p[0].m*p[2].m + sqr_p1/(4*mu13) - 2*p[0].m*p[1].m*R1/R;
	var_t c10 = 2*p[0].m*p[1].m*R1*R2/SQR(R);

	//Qdot (1-8)
	dy[0] = c1 * c4.x + c2 * p1.x;
	dy[1] = c1 * c4.y + c2 * p1.y;
	dy[2] = c1 * c4.z + c2 * p1.z;
	dy[3] = c1 * c4.w + c2 * p1.w;
	
	dy[4] = c1 * c5.x + c3 * p5.x;
	dy[5] = c1 * c5.y + c3 * p5.y;
	dy[6] = c1 * c5.z + c3 * p5.z;
	dy[7] = c1 * c5.w + c3 * p5.w;

	//Pdot (1-8)
	dy[8]  = c1 * c6.x + c8 * q1.x + c10 * (q1.x - q5.x); 
	dy[9]  = c1 * c6.y + c8 * q1.y + c10 * (q1.y - q5.y); 
	dy[10] = c1 * c6.z + c8 * q1.z + c10 * (q1.z - q5.z); 
	dy[11] = c1 * c6.w;
	
	dy[12] = c1 * c7.x + c9 * q5.x + c10 * (q1.x - q5.x); 
	dy[13] = c1 * c7.y + c9 * q5.y + c10 * (q1.y - q5.y); 
	dy[14] = c1 * c7.z + c9 * q5.z + c10 * (q1.z - q5.z); 
	dy[15] = c1 * c7.w;

	// dt = R1*R2*dtau
	dy[16] = R1*R2;

	//dy[0] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[0]*y_temp[4] + y_temp[1]*y_temp[5] + y_temp[2]*y_temp[6] ) + y_temp[13] * ( y_temp[1]*y_temp[4] - y_temp[0]*y_temp[5] + y_temp[2]*y_temp[7] ) - y_temp[14] * ( y_temp[0]*y_temp[6] - y_temp[2]*y_temp[4] + y_temp[1]*y_temp[7] ) + y_temp[15] * ( y_temp[0]*y_temp[7] - y_temp[1]*y_temp[6] + y_temp[2]*y_temp[5] ) ) + 0.25 / mu13 * y_temp[8] * R2; 
	//dy[1] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[0]*y_temp[5] - y_temp[1]*y_temp[4] + y_temp[3]*y_temp[6] ) + y_temp[13] * ( y_temp[0]*y_temp[4] + y_temp[1]*y_temp[5] + y_temp[3]*y_temp[7] ) + y_temp[14] * ( y_temp[1]*y_temp[6] - y_temp[0]*y_temp[7] + y_temp[3]*y_temp[4] ) - y_temp[15] * ( y_temp[0]*y_temp[6] + y_temp[1]*y_temp[7] - y_temp[3]*y_temp[5] ) ) + 0.25 / mu13 * y_temp[9] * R2; 
	//dy[2] = 0.25 / p[2].m * (-y_temp[12] * ( y_temp[2]*y_temp[4] - y_temp[0]*y_temp[6] + y_temp[3]*y_temp[5] ) + y_temp[13] * ( y_temp[0]*y_temp[7] + y_temp[2]*y_temp[5] - y_temp[3]*y_temp[4] ) + y_temp[14] * ( y_temp[0]*y_temp[4] - y_temp[2]*y_temp[6] + y_temp[3]*y_temp[7] ) + y_temp[15] * ( y_temp[0]*y_temp[5] - y_temp[2]*y_temp[7] + y_temp[3]*y_temp[6] ) ) + 0.25 / mu13 * y_temp[10] * R2;
	//dy[3] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[1]*y_temp[6] - y_temp[2]*y_temp[5] + y_temp[3]*y_temp[4] ) - y_temp[13] * ( y_temp[2]*y_temp[4] - y_temp[1]*y_temp[7] + y_temp[3]*y_temp[5] ) + y_temp[14] * ( y_temp[1]*y_temp[4] + y_temp[2]*y_temp[7] - y_temp[3]*y_temp[6] ) + y_temp[15] * ( y_temp[1]*y_temp[5] + y_temp[2]*y_temp[6] + y_temp[3]*y_temp[7] ) ) + 0.25 / mu13 * y_temp[11] * R2; 

	//dy[4] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) + y_temp[5] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) + y_temp[6] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) ) + 0.25 / mu23 * y_temp[12] * R1; 
	//dy[5] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) - y_temp[5] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) + y_temp[7] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) ) + 0.25 / mu23 * y_temp[13] * R1;
	//dy[6] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) - y_temp[6] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) - y_temp[7] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) ) + 0.25 / mu23 * y_temp[14] * R1;
	//dy[7] = 0.25 / p[2].m * ( y_temp[5] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) - y_temp[6] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) + y_temp[7] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) ) + 0.25 / mu23 * y_temp[15] * R1;
	
}

void threebody::gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void threebody::load(string& path)
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

void threebody::load_ascii(ifstream& input)
{
	threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	for (uint32_t i = 0; i < n_obj; i++)
	{
		load_ascii_record(input, &h_epoch[i], &h_md[i], &p[i], &h_y[i], &h_y[i+4]);
	}

	//load_ascii_record(input, &h_epoch[i], &h_md[i], &p[i], &h_y[i], &h_y[i+8]);
}

void threebody::load_ascii_record(ifstream& input, ttt_t* t, threebody_t::metadata_t *md, threebody_t::param_t* p, var_t* r, var_t* v)
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
	// mass
	input >> p->m;

	// position
	var4_t* _r = (var4_t*)r;
	input >> _r->x >> _r->y >> _r->z >> _r->w;

	// velocity
	var4_t* _v = (var4_t*)v;
	input >> _v->x >> _v->y >> _v->z >> _v->w;

}

void threebody::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void threebody::print_solution(std::string& path, data_rep_t repres)
{
	ofstream sout;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path.c_str(), ios::out | ios::app);
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

void threebody::print_solution_ascii(ofstream& sout)
{
	static uint32_t int_t_w  =  8;
	static uint32_t var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	for (uint32_t i = 0; i < n_obj; i++)
    {
		uint32_t orig_idx = h_md[i].id - 1;

		sout << setw(var_t_w) << t << SEP                       /* time of the record [day] (double)           */
			 << setw(     30) << obj_names[orig_idx] << SEP     /* name of the body         (string = 30 char) */ 
		// Print the metadata for each object
        << setw(int_t_w) << h_md[i].id << SEP;

		// Print the parameters for each object
		for (uint16_t j = 0; j < n_ppo; j++)
		{
			uint32_t param_idx = i * n_ppo + j;
			sout << setw(var_t_w) << h_p[param_idx] << SEP;
		}
		// Print the variables for each object
		for (uint16_t j = 0; j < n_vpo; j++)
		{
			uint32_t var_idx = i * n_vpo + j;
			sout << setw(var_t_w) << h_y[var_idx];
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

void threebody::print_solution_binary(ofstream& sout)
{
	throw string("The print_solution_binary() is not implemented.");
}

void threebody::print_integral(string& path)
{
	static uint32_t int_t_w  =  8;
	static uint32_t var_t_w  = 25;

	ofstream sout;

	sout.open(path.c_str(), ios::out | ios::app);
	if (sout)
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

	sout << setw(var_t_w) << t << SEP                       /* time of the record [day] (double)           */
		 << h << endl;
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	sout.close();
}
