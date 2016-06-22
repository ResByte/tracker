/*
 * parameters.hpp
 * Parameters for template learning and Scale filter are defined
 *
 * Copyright 2016 abhinav
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 *
 *
 */

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