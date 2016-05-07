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


#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

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
		int merge_method // 0 if const factor
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
	
	/* @brief: set Parameters */
	void setParameters();

	/* @brief: obtain image sub-window, padding is done by replicating border values.
	* Returns sub-window of image IM centered at POS ([y, x] coordinates),
	* with size MODEL_SZ ([height, width]). If any pixels are outside of the image,
	* they will replicate the values at the borders 
	*/
	virtual cv::Mat getSubWindow(cv::Mat& img, int x, int y, int height, int width);

	/* create new models for foreground and background or update the current ones */
	virtual void updateHistModel();
	
	void hann();

	params _parameter_config;

};

Tracker::Tracker()
{
	
}

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
