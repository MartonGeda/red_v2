#pragma once

#include "ode.h"

#include "type.h"

class threebody : public ode
{
public:
	threebody(uint16_t n_ppo, computing_device_t comp_dev);
	~threebody();

	void load(std::string& path);
	void load_ascii(std::ifstream& input);
	void load_ascii_record(std::ifstream& input, ttt_t* t, threebody_t::metadata_t *md, threebody_t::param_t* p, var_t* r, var_t* v);
	void load_binary(std::ifstream& input);

	//! Print the solution (the numerical approcimation of the solution)
	/*!
		\param path   full file path where the solution will be printed
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_solution(std::string& path, data_rep_t repres);
	//! Print the solution for each object in text format
	/*   
		\param sout print the data to this stream
	*/
	void print_solution_ascii(std::ofstream& sout);
	//! Print the solution for each object in binary format
	/*!
		\param sout print the data to this stream
	*/
	void print_solution_binary(std::ofstream& sout);

	//! Print the integral(s) of the problem
	/*!
		\param path   full file path where the integrals of the problem will be printed
	*/
	void print_integral(std::string& path);

	void calc_integral();
	void calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);

//	void trans_to_descartes(var3_t& q1, var3_t& p1, var3_t& q2, var3_t& p2, var3_t& q3, var3_t& p3, const var4_t& Q1, const var4_t& P1, const var4_t& Q2, const var4_t& P2);
//	void trans_to_threebody(const var3_t& qv1, const var3_t& pv1, const var3_t& qv2, const var3_t& pv2, const var3_t& qv3, const var3_t& pv3, var4_t& Q1, var4_t& P1, var4_t& Q2, var4_t& P2);

//private:
	void initialize();
	void allocate_storage();
	void allocate_host_storage();
	void allocate_device_storage();
	
	void deallocate_storage();
	void deallocate_host_storage();
	void deallocate_device_storage();

	void cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);
	void gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);

	var_t h;               //! Energy of the system

	threebody_t::metadata_t* h_md;
	threebody_t::metadata_t* d_md;
	threebody_t::metadata_t* md;

	var_t* h_epoch;
	var_t* d_epoch;
	var_t* epoch;
};