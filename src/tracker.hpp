/*
 * tracker.hpp
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

#include "image_processor.hpp"

class Tracker
{
public:	
	/* This is the main tracker class from STAPLE. 
	* 
	* 
	* TODO: 
	*/ 
	
	struct params{
	/* Parameters
	 * 
	 */
		bool grayscale_sequence;
		int hog_cell_size;
		int fixed_area;
		int n_bins;
		float learning_rate_pwp;
		int feature_type; // 0 if fhog , 1 if gray
		float inner_padding;
		float output_sigma_factor;
		float lambda;
		float learning_rate_cf;
		float merge_factor;
		int merge_method; // 0 if const factor
		bool den_per_channel;
		
		// scale related
		bool scale_adaptation;
		int hog_scale_cell_size;
		float learning_rate_scale;
		float scale_sigma_factor;
		int num_scales;
		float scale_model_factor;
		float scale_step;
		int scale_model_max_area;
		
	};


	/* @brief: Constructor which initializes  */
	Tracker(); 


};
