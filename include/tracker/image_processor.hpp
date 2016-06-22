/*
 * image_processor.hpp
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
#ifndef _IMAGE_PROCESSOR_HPP
#define _IMAGE_PROCESSOR_HPP

#include <iostream>
#include <string>
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

#include "boost/filesystem.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include "boost/foreach.hpp"
#include "boost/range.hpp"
#include "tracker/parameters.hpp"




class ImageProcessor
{
public:

	struct Position
	{
		int x;	// coordinate of the upper left most corner
		int y;
		int w;	// width of the frame
		int h;	// height of the frame
	};



	ImageProcessor(Parameters param, ScaleParameters s_param)
	{
		_fixed_patch_size = param.fixed_patch_size;
		_reg_param = param.lambda;
		_templ_learning_rate = param.templ_eta;

		// initialize scale filter parameters
		_scale_learning_rate = s_param.learning_rate;
		_scale_sigma_factor = s_param.sigma_factor;
		_num_scales = s_param._num_scales;
		_scale_model_factor = s_param.model_factor;
		_scale_step = s_param.scale_step;
		_scale_model_max_area = s_param.model_max_area;

		_p.x = 243;
		_p.y = 165;
		_p.w = 110;
		_p.h = 115;
	}

	// reads all file names in given directory
	void readDir();

	// reads image from file name
	// displays until the window is closed
	void readImage(std::string filename);

	//resize image to fixed size
	void resizeImg(cv::Mat& in, cv::Mat& out);

	// displays image untill shutdown
	void showImage(cv::Mat im);

	// show spectrum response image
	void showResponseImage(cv::Mat& img);

	// pre-process input image
	void preprocessImg(cv::Mat& img);


	// set current Image from file
	void setCurrentImage(std::string filename);

	// shows dft image 
	void showDFT(cv::Mat& complexImg);

	// compute hog features from given image
	void computeHoG(cv::Mat& img, cv::Mat& feature_image);

	// convolves two arrays in frequency domain using dft
	void convolveDFT(cv::Mat& A, cv::Mat& B, cv::Mat& output );

	// convolves 2 matrices in frquency domain and returns in frequency domain
	cv::Mat convolveDFTSpectrum(cv::Mat& A, cv::Mat& B);

	// get correlation filter for the image and patch
	void correlationFilter(cv::Mat& img, cv::Mat& filter, cv::Mat& output);

	// extract subimage from the given image
	void extractRect(cv::Mat& input, cv::Mat& output, int x, int y, int width, int height);

	// extracts a rectangular patch from the given image with pre-defined parameters
	cv::Mat extractPatch(cv::Mat& in, Position& p);

	// compute optimal correlation filter
	void getOptimalCorrelationFilter(cv::Mat input);

	// runs test algorithms
	void run();

	// initialize filter
	void initializeFilter(cv::Mat& y);

	// divides 2 spectrum in frequency domain
	void spectrumDiv(cv::Mat& a, cv::Mat& b, cv::Mat& out);

	

	// compute inverse of matrix having imaginery components
	void getComplexInverse(cv::Mat& in, cv::Mat& out);

	// generate training sample
	void createTrainingSample(std::vector<cv::Mat>& in, cv::Mat& sample);

	// computes DFT of given image
	void computeDFT(cv::Mat& in, cv::Mat& out);

	std::map<int, std::string> _data_map;
	Position _p;
	// template parameters
	float _reg_param;
	float _templ_learning_rate;
	int _fixed_patch_size; // every patch is resized to this before computing features
	//std::string _filename;

	// scale tracker parameters 
	double _scale_learning_rate;
	double _scale_sigma_factor;
	int _num_scales;
	double _scale_model_factor;
	double _scale_step;
	int _scale_model_max_area;

	cv::Mat _prev_image;
	cv::Mat _curr_image;
	cv::Mat _prev_roi;
	cv::Mat _curr_roi;
	cv::HOGDescriptor _hog;
};
#endif /* _IMAGE_PROCESSOR_HPP */
