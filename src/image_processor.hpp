#ifndef _IMAGE_PROCESSOR_HPP
#define _IMAGE_PROCESSOR_HPP
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
#include <boost/filesystem/convenience.hpp>
#include <boost/foreach.hpp>
#include <boost/range.hpp>

using namespace std;
using namespace cv;

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

	struct Target
	{
		int x;
		int y;
	};

	struct Scale
	{
		int w;
		int h;
	};

	struct ModelH
	{
		cv::Mat r; 	// r_hat in freq domain
		cv::Mat s;	// s_hat in freq domain
	};

	ImageProcessor()
	{
		_fixed_patch_size = 150;
		_reg_param = 0.01;
		_learning_rate = 0.025;
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

	// create gaussian kernel
	cv::Mat createGaussian();

	// displays image untill shutdown
	void showImage(cv::Mat im);

	// show spectrum response image
	void showResponseImage(cv::Mat& img);

	// pre-process input image
	void preprocessImg(cv::Mat& img);

	//sets filename
	void setFileName(std::string name);

	// set previous image
	void initializeImages(std::string filename);

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

	// computes model h
	void computeH(cv::Mat& img, ModelH& h_result);

	// compute inverse of matrix having imaginery components
	void getComplexInverse(cv::Mat& in, cv::Mat& out);

	// generate training sample
	void createTrainingSample(std::vector<cv::Mat>& in, cv::Mat& sample);

	// computes DFT of given image
	void computeDFT(cv::Mat& in, cv::Mat& out);

	std::map<int, string> _data_map;
	float _reg_param;
	float _learning_rate;
	Position _p;
	Target _prev_pos;
	Target _curr_pos;
	Scale _prev_scale;
	Scale _curr_scale;
	ModelH _prev_h;
	ModelH _curr_h;
	int _fixed_patch_size; // every patch is resized to this before computing features
	std::string _filename;
	cv::Mat _prev_image;
	cv::Mat _curr_image;
	cv::Mat _prev_roi;
	cv::Mat _curr_roi;
	cv::HOGDescriptor _hog;
};
#endif /* _IMAGE_PROCESSOR_HPP */
