/*
 * tracker.cpp
 * 
 * Copyright 2016 abhi <abhi@abhi-MacBookPro>
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

#include "tracker.hpp"

Tracker::Tracker()
{
	
}

/*
void Tracker::setParameters()
{
	_parameter_config.grayscale_sequence = false;	//suppose that sequence is colour
	_parameter_config.hog_cell_size = 4;
	_parameter_config.fixed_area = 150^2;          	//standard area to which we resize the target
	_parameter_config.n_bins = 2^5;             	//number of bins for the color histograms (bg and fg models)
	_parameter_config.learning_rate_pwp = 0.04;     //bg and fg color models learning rate 
	_parameter_config.feature_type = 1;
	_parameter_config.inner_padding = 0.2;          //defines inner area used to sample colors from the foreground
	_parameter_config.output_sigma_factor = 1/16 ;  //standard deviation for the desired translation filter output
	_parameter_config.lambda = 1e-3;                //regularization weight
	_parameter_config.learning_rate_cf = 0.01;      //HOG model learning rate
	_parameter_config.merge_factor = 0.3;           //fixed interpolation factor - how to linearly combine the two responses
	_parameter_config.merge_method = 1;
	_parameter_config.den_per_channel = false;

	// scale related
	_parameter_config.scale_adaptation = true;
	_parameter_config.hog_scale_cell_size = 4;      //Default DSST=4
	_parameter_config.learning_rate_scale = 0.025;
	_parameter_config.scale_sigma_factor = 1/4;
	_parameter_config.num_scales = 33;
	_parameter_config.scale_model_factor = 1.0;
	_parameter_config.scale_step = 1.02;
	_parameter_config.scale_model_max_area = 32*16;

	
}
*/	

int main(int argc, char **argv)
{
	/* code */
	Tracker track;
	return 0;
}