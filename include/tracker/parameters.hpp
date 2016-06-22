#ifndef _PARAMETERS_HPP
#define _PARAMETERS_HPP

#include <iostream>

struct Parameters
{
	int fixed_patch_size = 150;
	double lambda = static_cast<double>(0.01);
	double templ_eta = static_cast<double>(0.01);

};

struct ScaleParameters
{
	double learning_rate = static_cast<double>(0.025);
	double sigma_factor = static_cast<double>(1/4);
	int num_scales = 33;
	double model_factor = static_cast<double>(1.0);
	double scale_step = static_cast<double>(1.02);
	int model_max_area = 32*16;
	
};
#endif