#pragma once


#include "ode.h"

#include "type.h"

class nbody : public ode
{
public:
	nbody(uint16_t n_ppo, computing_device_t comp_dev);
	~nbody();

	void load(std::string& path);
	void load_ascii(ifstream& input);
	void load_ascii_record(ifstream& input, ttt_t* t, nbody_t::body_metadata_t* bmd, nbody_t::param_t* p, var_t* r, var_t* v);

};
