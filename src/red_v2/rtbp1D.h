#pragma once

#include "ode.h"
#include "type.h"

class rtbp1D : public ode
{
public:
	//! Constructs a rtbp1D object
	/*!
		\param n_ppo     the number of parameters per object
		\param comp_dev  the name of the executing device
	*/
	rtbp1D(uint16_t n_ppo, comp_dev_t comp_dev);
	//! Destructor
	~rtbp1D();

	//! Load the initial conditions from the hard disk
	/*!
		\param path   full file path of the file
	*/
	void load(std::string& path);

	//! Print the solution (the numerical approximation of the solution)
	/*!
		\param path_si  full file path where the info about the solution will be printed
		\param path_sd  full file path where the the solution will be printed
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_solution(std::string& path_si, std::string& path_sd, data_rep_t repres);

	//! Print the integral(s) of the problem
	/*!
		\param path   full file path where the integrals of the problem will be printed
	*/
	void print_integral(std::string& path);

	void calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);
	void calc_integral();

private:
	void initialize();
	void allocate_storage();
	void allocate_host_storage();
	void allocate_device_storage();

	void deallocate_storage();
	void deallocate_host_storage();
	void deallocate_device_storage();

	//! Load the initial conditions from the input stream (text mode)
	/*!
		\param input   the input stream associated with the file containing the initial conditions
	*/
	void load_ascii(std::ifstream& input);
	//! Load an initial condition record from the input stream (text mode)
	/*!
		\param input   the input stream associated with the file containing the initial conditions
		\param _t      the time for which the data are valid
		\param md      the metadata of the object
		\param p       the parameters of the object
		\param x       the x coordinate of the object
		\param vx      the vx velocity of the object
	*/
	void load_ascii_record(std::ifstream& input, ttt_t* _t, tbp1D_t::metadata_t *md, tbp1D_t::param_t* p, var_t* x, var_t* vx);
	//! Load the initial conditions from the input stream (binary mode)
	/*!
		\param input   the input stream associated with the file containing the initial conditions
	*/
	void load_binary(std::ifstream& input);
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

	void cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);
	void gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy);

	void trans_to_descartes_var(var_t& x, var_t& vx);

	tbp1D_t::integral_t integral;   //! Energy of the system
	tbp1D_t::metadata_t *h_md, *d_md, *md;
};
