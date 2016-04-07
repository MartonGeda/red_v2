#pragma once

#include "ode.h"

#include "type.h"

class tbp3D : public ode
{
public:
	tbp3D(uint16_t n_ppo, computing_device_t comp_dev);
	~tbp3D();

	void load(std::string& path);
	void load_ascii(std::ifstream& input);
	void load_ascii_record(std::ifstream& input, ttt_t* t, tbp3D_t::metadata_t *md, tbp3D_t::param_t* p, var_t* r, var_t* v);
	void load_binary(std::ifstream& input);

	//! Print the solution (the numerical approcimation of the solution)
	/*!
		\param path   full file path where the solution will be printed
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_solution(std::string& path, data_rep_t repres);
	//! Print the solution for each object in binary format
	/*   
		\param sout print the data to this stream
	*/
	void print_solution_ascii(std::ofstream& sout);
	//! Print the data of all bodies in binary format
	/*!
		\param sout print the data to this stream
	*/
	void print_solution_binary(std::ofstream& sout);

	//! Print the integral(s) of the problem
	/*!
		\param path   full file path where the integrals of the problem will be printed
	*/
	void print_integral(std::string& path);

	void calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);
	void calc_integral();

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

	tbp3D_t::metadata_t* h_md;
	tbp3D_t::metadata_t* d_md;
	tbp3D_t::metadata_t* md;

	var_t* h_epoch;
	var_t* d_epoch;
	var_t* epoch;
};